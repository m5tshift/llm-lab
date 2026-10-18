"""
Abstract base interfaces (ports) for multimodal processing pipelines.
Enables loose coupling, test mocking, and clean dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Any, List
from src.core.schemas import (
    SingleSentimentResult,
    BatchSentimentResult,
    AudioTranscriptionResult,
    ImageClassificationResult,
    VideoDetectionResult,
    LLMReplyResult,
)


class BaseTextAnalyzer(ABC):
    """Port for text sentiment classification."""

    @abstractmethod
    def analyze_message(self, text: str) -> SingleSentimentResult:
        pass

    @abstractmethod
    def analyze_chat_batch(self, messages: List[str]) -> BatchSentimentResult:
        pass


class BaseAudioTranscriber(ABC):
    """Port for speech-to-text audio transcription."""

    @abstractmethod
    def transcribe_audio_file(self, audio_path: str) -> AudioTranscriptionResult:
        pass


class BaseImageClassifier(ABC):
    """Port for single-image classification."""

    @abstractmethod
    def classify_image(self, image_input: Any) -> ImageClassificationResult:
        pass


class BaseVideoDetector(ABC):
    """Port for video frame object detection."""

    @abstractmethod
    def process_frame(self, frame: Any) -> VideoDetectionResult:
        pass


class BaseLLMBrain(ABC):
    """Port for local or remote LLM reaction generation."""

    @abstractmethod
    def generate_reply(
        self, streamer_speech: str, chat_mood: str, visual_context: str
    ) -> LLMReplyResult:
        pass


# Aliases for I-prefixed interface style
ITextSentimentAnalyzer = BaseTextAnalyzer
IAudioTranscriber = BaseAudioTranscriber
IImageClassifier = BaseImageClassifier
IVideoDetector = BaseVideoDetector
ILLMBrain = BaseLLMBrain
