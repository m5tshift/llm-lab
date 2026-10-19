"""Module 2: Audio Processing - Streamer Speech-to-Text (STT).

Framework: Hugging Face (transformers) / OpenAI Whisper
Task: Automatic Speech Recognition (ASR)
"""

import time
import os
import re
from typing import Any, Dict, Optional, Union
from collections import Counter
from transformers import pipeline

from src.core.config import settings
from src.core.schemas import AudioTranscriptionResult
from src.core.interfaces import BaseAudioTranscriber

# Common Whisper hallucination phrases on silence/music in Russian & English
KNOWN_HALLUCINATIONS = [
    "редактор субтитров",
    "корректор а.кулакова",
    "а.кулакова",
    "корректор",
    "субтитры сделал",
    "субтитры подготовил",
    "перевод субтитров",
    "переведено и озвучено",
    "озвучено",
    "озвучил",
    "спасибо за просмотр",
    "ставьте лайки",
    "подписывайтесь на канал",
    "продолжение следует",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
]


def clean_transcription_text(text: str) -> str:
    """Cleans raw Whisper output and removes typical silence/music hallucinations."""
    if not text:
        return ""
    cleaned = text.strip()
    lowered = cleaned.lower()

    # 1. Iteratively remove all known subtitle credits hallucinations
    rem = lowered
    for phrase in KNOWN_HALLUCINATIONS:
        if phrase in rem:
            rem = rem.replace(phrase, "")
    rem = rem.strip(" .,!-:;()[]")
    if len(rem) < 6:
        return ""

    # 2. Check for repetitive hallucination loops (e.g. 'Да. Да. Да. Да.')
    words = re.findall(r"\w+", lowered)
    if len(words) >= 5:
        counts = Counter(words)
        _, top_count = counts.most_common(1)[0]
        if top_count / len(words) > 0.45:
            return ""

    # 3. Strip non-informative single punctuation
    if len(re.sub(r"[^a-zA-Zа-яА-Я0-9]", "", cleaned)) < 2:
        return ""

    return cleaned


class StreamAudioTranscriber(BaseAudioTranscriber):
    """Transcribes audio clips into text using OpenAI Whisper."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_language: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 64,
    ):
        self.model_name = model_name or settings.models.whisper_model
        self.default_language = default_language or settings.models.whisper_language
        self.device = device or "cpu"
        # 64 tokens fit the 3-5 s live chunks; longer evaluation utterances need more.
        self.max_new_tokens = max_new_tokens

        print(
            f"[Audio] Loading ASR Whisper model: '{self.model_name}' (Default language: {self.default_language})..."
        )
        start = time.perf_counter()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            chunk_length_s=30,
            device=self.device,
        )
        self.load_time = time.perf_counter() - start
        print(f"[Audio] Model loaded successfully in {self.load_time:.2f}s")

    def transcribe_audio_file(
        self, audio_path: str, language: Optional[str] = None
    ) -> AudioTranscriptionResult:
        """Transcribes an audio file (.wav, .mp3, .flac)."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self.transcribe(audio_path, language=language)

    def transcribe(
        self, audio: Union[str, Dict[str, Any]], language: Optional[str] = None
    ) -> AudioTranscriptionResult:
        """Transcribes a file path or {"array": np.ndarray, "sampling_rate": int}."""
        if isinstance(audio, dict):
            audio = {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}

        target_lang = language if language is not None else self.default_language
        gen_kwargs = {
            "task": "transcribe",
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": 3,
        }
        if target_lang and target_lang.lower() not in ("auto", "none", "detect"):
            gen_kwargs["language"] = target_lang.lower()

        start = time.perf_counter()
        result = self.pipe(audio, generate_kwargs=gen_kwargs)
        inference_time_s = time.perf_counter() - start

        raw_text = result.get("text", "").strip()
        cleaned_text = clean_transcription_text(raw_text)

        return AudioTranscriptionResult(
            transcription=cleaned_text,
            raw_transcription=raw_text,
            language=target_lang or "auto",
            inference_time_s=round(inference_time_s, 3),
        )


if __name__ == "__main__":
    import wave
    import numpy as np

    print("--- Testing Stream Audio Transcriber ---")
    transcriber = StreamAudioTranscriber()

    test_wav = "sample_test_speech.wav"
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(test_wav, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    try:
        res = transcriber.transcribe_audio_file(test_wav)
        print("Transcription result on tone/silence:", res)
    finally:
        if os.path.exists(test_wav):
            os.remove(test_wav)
