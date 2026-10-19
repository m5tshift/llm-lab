"""Shared Web UI application state and model registry."""

from typing import Optional, Dict, Any

from src.nlp_sentiment import ChatSentimentAnalyzer
from src.audio_transcribe import StreamAudioTranscriber
from src.scene_classify import SceneClassifier
from src.video_live import LiveStreamVideoDetector
from src.llm_brain import LocalLLMBrain
from src.vision_backend import OllamaVisionClassifier


class WebState:
    """Encapsulates active models and live stream context."""

    def __init__(self):
        self.stream_state: Dict[str, Any] = {
            "channel": "",
            "speech": "",
            "speech_history": [],
            "visual": "",
            "scene": "",
            "vlm": "",
            "mood": "",
            "chat_stats": {},
            "chat_messages": [],
            "fps": 0.0,
            "session_time": 0,
            "is_live": False,
        }
        self.active_chat_listener: Any = None
        self.active_audio_listener: Any = None
        self.nlp_model: Optional[ChatSentimentAnalyzer] = None
        self.audio_model: Optional[StreamAudioTranscriber] = None
        self.scene_model: Optional[SceneClassifier] = None
        self.vision_model: Optional[OllamaVisionClassifier] = None
        self.video_model: Optional[LiveStreamVideoDetector] = None
        self.llm_model: Optional[LocalLLMBrain] = None

    def preload_models(self):
        """Loads models safely with individual exception handling."""
        print("[Web UI] Pre-loading models for interactive demo...")
        try:
            self.nlp_model = ChatSentimentAnalyzer()
        except Exception as e:
            print(f"[Warn] NLP model load deferred: {e}")

        try:
            self.audio_model = StreamAudioTranscriber()
        except Exception as e:
            print(f"[Warn] Audio model load deferred: {e}")

        try:
            self.scene_model = SceneClassifier()
        except Exception as e:
            print(f"[Warn] Scene model load deferred: {e}")

        try:
            self.vision_model = OllamaVisionClassifier()
        except Exception as e:
            print(f"[Warn] Vision model load deferred: {e}")

        try:
            self.video_model = LiveStreamVideoDetector()
        except Exception as e:
            print(f"[Warn] Video model load deferred: {e}")

        try:
            self.llm_model = LocalLLMBrain()
        except Exception as e:
            print(f"[Warn] LLM model load deferred: {e}")


web_state = WebState()
