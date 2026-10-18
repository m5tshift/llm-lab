"""Unit and integration tests for FastAPI service endpoints."""

import unittest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.api import app
import src.api as api_module
from src.core.schemas import (
    SingleSentimentResult,
    BatchSentimentResult,
    ImageClassificationResult,
    LLMReplyResult,
)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("/metrics", data.get("endpoints", []))

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"python_info", resp.content)

    def test_single_sentiment_endpoint(self):
        mock_nlp = MagicMock()
        mock_nlp.analyze_message.return_value = SingleSentimentResult(
            text="awesome match", label="POSITIVE", score=0.98, latency_ms=15.0
        )
        api_module.nlp_model = mock_nlp

        resp = self.client.post("/api/text/sentiment", json={"text": "awesome match"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["label"], "POSITIVE")
        self.assertAlmostEqual(data["score"], 0.98)

    def test_batch_sentiment_endpoint(self):
        mock_nlp = MagicMock()
        mock_nlp.analyze_chat_batch.return_value = BatchSentimentResult(
            dominant_mood="POSITIVE",
            positive_ratio=0.85,
            negative_ratio=0.05,
            total_messages=10,
            latency_ms=25.0,
        )
        api_module.nlp_model = mock_nlp

        resp = self.client.post(
            "/api/text/batch-sentiment", json={"messages": ["gg", "wp", "let's go"]}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["dominant_mood"], "POSITIVE")
        self.assertEqual(data["total_messages"], 10)

    def test_llm_reaction_endpoint(self):
        mock_llm = MagicMock()
        mock_llm.generate_reply.return_value = LLMReplyResult(
            reply="Bot says: GL HF!", engine="TestEngine", latency_ms=18.0
        )
        api_module.llm_model = mock_llm

        resp = self.client.post(
            "/api/llm/generate-reaction",
            json={
                "streamer_speech": "Let's push A site",
                "chat_mood": "HYPE",
                "visual_context": "gameplay screen",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["reply"], "Bot says: GL HF!")
        self.assertEqual(data["engine"], "TestEngine")

    def test_image_vlm_endpoint(self):
        mock_image = MagicMock()
        mock_image.classify_image.return_value = ImageClassificationResult(
            top_prediction="На экране трансляции виден интерфейс игры, счет 5:3",
            confidence=0.98,
            predictions=[],
            inference_latency_ms=120.0,
        )
        api_module.vlm_model = mock_image

        file_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        resp = self.client.post(
            "/api/image/vlm",
            files={"file": ("test.png", file_content, "image/png")},
            data={"prompt": "Кратко на русском опиши кадр"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            data["top_prediction"], "На экране трансляции виден интерфейс игры, счет 5:3"
        )
        self.assertEqual(data["confidence"], 0.98)


if __name__ == "__main__":
    unittest.main()
