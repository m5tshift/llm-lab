"""Unit tests for centralized configuration."""

import unittest
import os
from src.core.config import AppConfig


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = AppConfig()
        self.assertIsNotNone(cfg.models.nlp_model)
        self.assertIsNotNone(cfg.models.whisper_model)
        self.assertIsNotNone(cfg.models.vlm_model)
        self.assertIsNotNone(cfg.network.api_port)
        self.assertEqual(cfg.twitch.irc_port, 6697)

    def test_env_override(self):
        os.environ["DEFAULT_TWITCH_CHANNEL"] = "testchannel123"
        cfg = AppConfig()
        self.assertEqual(cfg.twitch.default_channel, "testchannel123")


if __name__ == "__main__":
    unittest.main()
