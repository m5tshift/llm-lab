"""Unit tests for Whisper hallucination cleaning logic."""

import unittest
from src.audio_transcribe import clean_transcription_text


class TestAudioCleaning(unittest.TestCase):
    def test_clean_valid_speech(self):
        text = "Всем привет, начинаем наш стрим по CS!"
        cleaned = clean_transcription_text(text)
        self.assertEqual(cleaned, text)

    def test_remove_hallucination_subtitle_credits(self):
        text = "Субтитры подготовил и озвучил а.кулакова"
        cleaned = clean_transcription_text(text)
        self.assertEqual(cleaned, "")

    def test_remove_repetitive_loop(self):
        text = "Да. Да. Да. Да. Да. Да. Да."
        cleaned = clean_transcription_text(text)
        self.assertEqual(cleaned, "")

    def test_remove_empty_or_punctuation_only(self):
        self.assertEqual(clean_transcription_text(""), "")
        self.assertEqual(clean_transcription_text("..."), "")
        self.assertEqual(clean_transcription_text("   "), "")


if __name__ == "__main__":
    unittest.main()
