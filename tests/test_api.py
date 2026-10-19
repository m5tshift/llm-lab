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
        api_module.vision_model = mock_image

        file_content = _png_bytes()
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

    def test_scene_endpoint_returns_class_probabilities(self):
        mock_scene = MagicMock()
        mock_scene.classify_image.return_value = ImageClassificationResult(
            top_prediction="gameplay", confidence=0.91, predictions=[], inference_latency_ms=30.0
        )
        api_module.scene_model = mock_scene
        resp = self.client.post(
            "/api/image/scene", files={"file": ("frame.png", _png_bytes(), "image/png")}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["top_prediction"], "gameplay")

    def test_non_image_upload_is_rejected_with_400(self):
        api_module.scene_model = MagicMock()
        resp = self.client.post(
            "/api/image/scene", files={"file": ("notes.txt", b"not an image", "text/plain")}
        )
        self.assertEqual(resp.status_code, 400)
        api_module.scene_model.classify_image.assert_not_called()

    def test_empty_batch_is_rejected(self):
        api_module.nlp_model = MagicMock()
        resp = self.client.post("/api/text/batch-sentiment", json={"messages": []})
        self.assertEqual(resp.status_code, 422)


def _png_bytes():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
