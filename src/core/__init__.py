"""Core module providing settings, schemas, and abstract interfaces."""

from src.core.config import settings, AppConfig
from src.core.schemas import (
    TwitchChatMessage,
    SingleSentimentResult,
    SentimentDetail,
    BatchSentimentResult,
    AudioTranscriptionResult,
    ImageClassificationResult,
    ClassificationItem,
    DetectedObject,
    VideoDetectionResult,
    LLMReplyResult,
    MultimodalCycleResult,
)
from src.core.interfaces import (
    BaseTextAnalyzer,
    BaseAudioTranscriber,
    BaseImageClassifier,
    BaseVideoDetector,
    BaseLLMBrain,
    ITextSentimentAnalyzer,
    IAudioTranscriber,
    IImageClassifier,
    IVideoDetector,
    ILLMBrain,
)

__all__ = [
    "settings",
    "AppConfig",
    "TwitchChatMessage",
    "SingleSentimentResult",
    "SentimentDetail",
    "BatchSentimentResult",
    "AudioTranscriptionResult",
    "ImageClassificationResult",
    "ClassificationItem",
    "DetectedObject",
    "VideoDetectionResult",
    "LLMReplyResult",
    "MultimodalCycleResult",
    "BaseTextAnalyzer",
    "BaseAudioTranscriber",
    "BaseImageClassifier",
    "BaseVideoDetector",
    "BaseLLMBrain",
    "ITextSentimentAnalyzer",
    "IAudioTranscriber",
    "IImageClassifier",
    "IVideoDetector",
    "ILLMBrain",
]
