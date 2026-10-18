"""Non-blocking, periodic VLM analysis of live stream frames."""

import threading
import time
from typing import Callable, Optional

from src.vision_backend import create_vision_classifier
from src.metrics import record_model_latency, record_processing_error, VLM_FRAME_TIMESTAMP
from src.monitoring.decorators import safe_telemetry_dispatch


LIVE_VLM_PROMPT = (
    "Кратко на русском опиши, что происходит на кадре стрима. "
    "Укажи игру, видимые действия, важные элементы HUD и читаемый текст или счёт. "
    "Если текст неразборчив, не угадывай."
)


class BackgroundVLMAnalyzer:
    """Keeps one recent visual result without delaying the stream loop."""

    def __init__(
        self,
        model_factory: Optional[Callable] = None,
        interval_seconds: float = 20,
        max_age_seconds: float = 60,
    ):
        self.model_factory = model_factory or create_vision_classifier
        self.interval_seconds = max(0, interval_seconds)
        self.max_age_seconds = max_age_seconds
        self._lock = threading.Lock()
        self._model = None
        self._busy = False
        self._closed = False
        self._last_started = float("-inf")
        self._description = ""
        self._updated_at = 0.0
        self._error = ""
        self._frame_timestamp = 0.0
        safe_telemetry_dispatch(VLM_FRAME_TIMESTAMP.set, float("nan"))

    def submit(self, frame) -> bool:
        """Schedules a copied frame if the worker is idle and due."""
        now = time.monotonic()
        with self._lock:
            if self._closed or self._busy or now - self._last_started < self.interval_seconds:
                return False
            self._busy = True
            self._last_started = now
            self._frame_timestamp = time.time()
        try:
            snapshot = frame.copy()
            threading.Thread(target=self._analyze, args=(snapshot,), daemon=True).start()
        except Exception:
            with self._lock:
                self._busy = False
            raise
        return True

    def _analyze(self, frame) -> None:
        try:
            if self._model is None:
                self._model = self.model_factory()
            result = self._model.classify_image(frame, prompt=LIVE_VLM_PROMPT, max_new_tokens=80)
            description = result.top_prediction.strip()
            if not description:
                raise ValueError("Empty visual description")
            safe_telemetry_dispatch(
                record_model_latency, "image_vlm", result.inference_latency_ms / 1000.0
            )
            with self._lock:
                if not self._closed:
                    self._description = description
                    self._updated_at = self._last_started
                    self._error = ""
                    safe_telemetry_dispatch(VLM_FRAME_TIMESTAMP.set, self._frame_timestamp)
        except Exception as exc:
            safe_telemetry_dispatch(record_processing_error, "image_vlm")
            with self._lock:
                self._error = str(exc)
            print(f"[Live VLM] Analysis failed: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def get_latest(self) -> str:
        """Returns only a completed, fresh analysis."""
        with self._lock:
            if time.monotonic() - self._updated_at <= self.max_age_seconds:
                return self._description
            return ""

    def get_status(self) -> str:
        with self._lock:
            if self._description and time.monotonic() - self._updated_at <= self.max_age_seconds:
                return self._description
            if self._busy:
                return "Модель анализирует кадр..."
            if self._error:
                return f"Ошибка анализа кадра ({self._error})"
            return "Ожидание свежего анализа кадра..."

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._description = ""
            safe_telemetry_dispatch(VLM_FRAME_TIMESTAMP.set, float("nan"))
