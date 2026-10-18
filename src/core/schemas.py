"""
Domain schemas and Data Transfer Objects (DTOs) for Multimodal Stream Analyzer.
Provides typed contracts across pipeline layers with dual dictionary/attribute access.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SubscriptableModel(BaseModel):
    """Pydantic model supporting dictionary-like item access obj['key'] and obj.get('key')."""

    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TwitchChatMessage(SubscriptableModel):
    """Twitch IRC chat message DTO."""

    sender: str = Field(..., description="Username of the message author")
    text: str = Field(..., description="Message text content")
    color: str = Field(default="#1E90FF", description="User badge / nickname hex color")
    timestamp: str = Field(default="", description="Message arrival time (HH:MM:SS)")


class SingleSentimentResult(SubscriptableModel):
    """Sentiment and emotion classification result for a single chat line."""

    text: str
    label: str
    score: float
    latency_ms: float
    dominant_emotion: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None


class SentimentDetail(SubscriptableModel):
    """Individual message sentiment item inside a batch."""

    message: str
    label: str
    score: float
    dominant_emotion: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None


class BatchSentimentResult(SubscriptableModel):
    """Aggregated sentiment and emotional analysis for a window of chat messages."""

    dominant_mood: str = "NEUTRAL"
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0
    total_messages: int = 0
    latency_ms: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    avg_latency_per_message_ms: float = 0.0
    dominant_emotion: Optional[str] = None
    detailed: List[SentimentDetail] = Field(default_factory=list)


class AudioTranscriptionResult(SubscriptableModel):
    """Speech-to-text inference result from Faster-Whisper."""

    transcription: str
    language: str
    inference_time_s: float
    cleaned_text: Optional[str] = None


class ClassificationItem(SubscriptableModel):
    """Individual ranked ImageNet prediction."""

    rank: int
    label: str
    probability: float
    percentage: str


class ImageClassificationResult(SubscriptableModel):
    """Image classification result for a camera frame or screenshot."""

    top_prediction: str
    confidence: float
    predictions: List[ClassificationItem] = Field(default_factory=list)
    inference_latency_ms: float


class DetectedObject(SubscriptableModel):
    """Single YOLO detected bounding box item."""

    class_name: str
    confidence: float
    box: List[Union[int, float]] = Field(default_factory=list)


class VideoDetectionResult(SubscriptableModel):
    """Frame object detection result from YOLOv8."""

    detected_objects: List[DetectedObject] = Field(default_factory=list)
    annotated_frame: Any = None
    latency_ms: float = 0.0


class LLMReplyResult(SubscriptableModel):
    """Generated reaction from local LLM brain."""

    reply: str
    engine: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    speed_tps: float = 0.0


class MultimodalCycleResult(SubscriptableModel):
    """Complete multimodal monitoring cycle output combining all sensory streams."""

    streamer_speech: str
    chat_mood: str
    visual_objects: List[str]
    scene_classification: str
    bot_chat_reply: str
    llm_engine: str
    cycle_total_latency_ms: float
    detailed_results: Optional[Dict[str, Any]] = None
