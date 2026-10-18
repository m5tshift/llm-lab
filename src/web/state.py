"""Shared Web UI application state and model registry."""

from typing import Optional, Dict, Any
import threading

from src.nlp_sentiment import ChatSentimentAnalyzer
from src.audio_transcribe import StreamAudioTranscriber
from src.image_classify import StreamImageClassifier
from src.video_live import LiveStreamVideoDetector
from src.llm_brain import LocalLLMBrain
from src.vision_backend import create_vision_classifier


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
        self.image_model: Optional[StreamImageClassifier] = None
        self._image_models: Dict[str, StreamImageClassifier] = {}
        self._image_models_lock = threading.Lock()
        self.vision_model = None
        self.video_model: Optional[LiveStreamVideoDetector] = None
        self.llm_model: Optional[LocalLLMBrain] = None

    def get_image_classifier(self, model_name: Optional[str] = None) -> StreamImageClassifier:
        """Returns loaded image/VLM classifier from cache or instantiates it on-demand."""
        from src.core.config import settings

        target_name = (
            model_name or getattr(settings.models, "vlm_model", "Qwen/Qwen2-VL-2B-Instruct")
        ).strip()
        with self._image_models_lock:
            if target_name not in self._image_models:
                print(f"[Web UI] Initializing vision/VLM model: {target_name}...")
                self._image_models[target_name] = StreamImageClassifier(model_name=target_name)
            return self._image_models[target_name]

    def preload_models(self):
        """Loads models safely with individual exception handling."""
        print("[Web UI] Pre-loading models for interactive demo...")
        from src.core.config import settings

        try:
            self.nlp_model = ChatSentimentAnalyzer()
        except Exception as e:
            print(f"[Warn] NLP model load deferred: {e}")

        try:
            self.audio_model = StreamAudioTranscriber()
        except Exception as e:
            print(f"[Warn] Audio model load deferred: {e}")

        try:
            self.vision_model = create_vision_classifier()
            if settings.models.vision_backend == "transformers":
                self.image_model = self.get_image_classifier(settings.models.image_model)
            else:
                self.image_model = None
        except Exception as e:
            print(f"[Warn] Image model load deferred: {e}")

        try:
            self.video_model = LiveStreamVideoDetector()
        except Exception as e:
            print(f"[Warn] Video model load deferred: {e}")

        try:
            self.llm_model = LocalLLMBrain()
        except Exception as e:
            print(f"[Warn] LLM model load deferred: {e}")


web_state = WebState()
