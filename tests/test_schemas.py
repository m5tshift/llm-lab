"""Unit tests for domain schemas and DTO contracts."""

import unittest
from src.core.schemas import TwitchChatMessage, SingleSentimentResult, MultimodalCycleResult


class TestSchemas(unittest.TestCase):
    def test_twitch_chat_message(self):
        msg = TwitchChatMessage(
            sender="pogchamp_user", text="hello world", color="#FF0000", timestamp="12:00:00"
        )
        self.assertEqual(msg.sender, "pogchamp_user")
        self.assertEqual(msg["sender"], "pogchamp_user")
        self.assertEqual(msg.get("color"), "#FF0000")

    def test_sentiment_result_dual_access(self):
        res = SingleSentimentResult(
            text="great stream!", label="POSITIVE", score=0.985, latency_ms=12.4
        )
        self.assertEqual(res.label, "POSITIVE")
        self.assertEqual(res["label"], "POSITIVE")
        self.assertEqual(res.score, 0.985)
        self.assertEqual(res.get("latency_ms"), 12.4)

    def test_multimodal_cycle_result(self):
        cycle = MultimodalCycleResult(
            streamer_speech="Going for the plant",
            chat_mood="HYPE",
            visual_objects=["person", "laptop"],
            scene_classification="computer screen",
            bot_chat_reply="Nice play!",
            llm_engine="Ollama",
            cycle_total_latency_ms=45.2,
        )
        self.assertEqual(cycle.streamer_speech, "Going for the plant")
        self.assertEqual(cycle["bot_chat_reply"], "Nice play!")
        self.assertIn("person", cycle.visual_objects)


if __name__ == "__main__":
    unittest.main()
