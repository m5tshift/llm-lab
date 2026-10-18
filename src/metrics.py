"""Operational telemetry. No estimated quality scores or hardware upgrade predictions."""

import json
import threading
import time
import urllib.request
from collections import deque

import psutil
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST as CONTENT_TYPE_LATEST

# Keep bounded labels: components/model tags, never message text, channels or URLs.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160)
MODEL_LATENCY = Histogram(
    "twitch_model_latency_seconds",
    "Successful inference duration",
    ["model"],
    buckets=LATENCY_BUCKETS,
)
MODEL_LATEST_LATENCY = Gauge(
    "twitch_model_latest_latency_ms", "Latest successful inference duration in ms", ["model"]
)
PROCESSING_ERRORS = Counter(
    "twitch_processing_errors_total", "Failed processing operations", ["component"]
)
AI_UPDATE_DURATION = Histogram(
    "twitch_ai_update_duration_seconds",
    "AI worker duration, excluding parallel video/audio/VLM workers",
    ["mode"],
    buckets=LATENCY_BUCKETS,
)
FRAMES_PROCESSED = Counter(
    "twitch_frames_processed_total", "Frames processed by the application (not source FPS)"
)
STREAM_ACTIVE = Gauge("twitch_stream_active", "Whether a live session is running")
SESSION_DURATION_SECONDS = Gauge(
    "twitch_session_duration_seconds", "Elapsed time of the active session"
)
CHAT_RECEIVED = Counter(
    "twitch_chat_received_messages_total", "IRC messages received, independent of repeated analysis"
)
CHAT_POSITIVE_RATIO = Gauge(
    "twitch_chat_positive_ratio", "Positive fraction in the latest analyzed message window"
)
CHAT_NEGATIVE_RATIO = Gauge(
    "twitch_chat_negative_ratio", "Negative fraction in the latest analyzed message window"
)
VIDEO_LATEST_OBJECTS_COUNT = Gauge(
    "twitch_video_latest_objects_count",
    "Detections in the latest frame, not unique tracked objects",
)
BOT_REPLIES_TOTAL = Counter(
    "twitch_bot_replies_total", "Generated replies (not messages sent to Twitch)", ["engine"]
)
API_REQUESTS_TOTAL = Counter(
    "twitch_api_requests_total",
    "HTTP requests excluding health and metrics",
    ["endpoint", "method", "status_code"],
)
API_REQUEST_DURATION = Histogram(
    "twitch_api_request_duration_seconds",
    "HTTP request duration",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
OLLAMA_REQUESTS = Counter(
    "twitch_ollama_requests_total",
    "Ollama generation attempts including queue failures",
    ["model", "task", "status"],
)
LLM_TOKENS = Counter(
    "twitch_llm_tokens_total", "Tokens reported by Ollama", ["model", "task", "direction"]
)
LLM_GENERATION_SPEED = Gauge(
    "twitch_llm_generation_speed_tokens_per_second",
    "Latest output tokens divided by generation time",
    ["model", "task"],
)
OLLAMA_UP = Gauge("twitch_ollama_up", "Whether the latest Ollama runtime query succeeded")
OLLAMA_MODEL_LOADED = Gauge(
    "twitch_ollama_model_loaded", "Model resident in Ollama, possibly partly in RAM", ["model"]
)
OLLAMA_VRAM_USAGE_BYTES = Gauge(
    "twitch_ollama_vram_usage_bytes", "VRAM allocated by Ollama, not total GPU usage", ["model"]
)
VLM_FRAME_TIMESTAMP = Gauge(
    "twitch_vlm_frame_timestamp_seconds",
    "Capture/submission timestamp of the latest successful VLM frame; NaN when unavailable",
)
PROCESS_THREADS_COUNT = Gauge(
    "twitch_process_threads_count", "OS threads in this application process"
)
# RSS and CPU already exist as process_resident_memory_bytes/process_cpu_seconds_total.
VLM_FRAME_TIMESTAMP.set(float("nan"))
CHAT_POSITIVE_RATIO.set(float("nan"))
CHAT_NEGATIVE_RATIO.set(float("nan"))


def record_model_latency(model_name, latency_seconds):
    model_name = {"video_detector": "video_yolo"}.get(model_name, model_name)
    MODEL_LATENCY.labels(model=model_name).observe(latency_seconds)
    MODEL_LATEST_LATENCY.labels(model=model_name).set(latency_seconds * 1000)


def record_processing_error(component):
    PROCESSING_ERRORS.labels(component=component).inc()


def record_ai_update(duration, mode="live"):
    AI_UPDATE_DURATION.labels(mode=mode).observe(duration)


def record_chat_stats(positive_ratio, negative_ratio):
    CHAT_POSITIVE_RATIO.set(positive_ratio)
    CHAT_NEGATIVE_RATIO.set(negative_ratio)


def record_chat_received():
    CHAT_RECEIVED.inc()


def record_detected_objects(objects):
    VIDEO_LATEST_OBJECTS_COUNT.set(len(objects))


def record_bot_reply(engine="local"):
    BOT_REPLIES_TOTAL.labels(engine=engine).inc()


def record_frames_processed(count=1):
    FRAMES_PROCESSED.inc(count)


class FrameRateTracker:
    """Wall-clock throughput over a sliding window, including UI/network waits."""

    def __init__(self, window_seconds=5, clock=time.monotonic):
        self.clock = clock
        self.window = window_seconds
        self.started = clock()
        self.frames = deque()

    def tick(self):
        now = self.clock()
        self.frames.append(now)
        while self.frames and self.frames[0] <= now - self.window:
            self.frames.popleft()
        elapsed = min(self.window, now - self.started)
        return len(self.frames) / elapsed if elapsed > 0 else 0.0


_session_start_time = None
_session_lock = threading.Lock()


def set_stream_active(active):
    global _session_start_time
    with _session_lock:
        STREAM_ACTIVE.set(int(active))
        if active and _session_start_time is None:
            _session_start_time = time.monotonic()
        if not active:
            _session_start_time = None
            SESSION_DURATION_SECONDS.set(0)
            CHAT_POSITIVE_RATIO.set(float("nan"))
            CHAT_NEGATIVE_RATIO.set(float("nan"))
            VIDEO_LATEST_OBJECTS_COUNT.set(float("nan"))


def update_session_duration():
    with _session_lock:
        if _session_start_time is not None:
            SESSION_DURATION_SECONDS.set(time.monotonic() - _session_start_time)


def record_llm_tokens(
    prompt_tokens, completion_tokens, eval_duration_ns=0, model="unknown", task="text"
):
    for direction, count in (("input", prompt_tokens), ("output", completion_tokens)):
        LLM_TOKENS.labels(model=model, task=task, direction=direction).inc(max(0, count))
    speed = completion_tokens / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else float("nan")
    LLM_GENERATION_SPEED.labels(model=model, task=task).set(speed)


_process = psutil.Process()


def update_system_resource_metrics():
    """Small UI snapshot. Prometheus uses the standard process collector for CPU/RSS."""
    threads = _process.num_threads()
    PROCESS_THREADS_COUNT.set(threads)
    return {
        "process_memory_mb": round(_process.memory_info().rss / 1024**2, 1),
        "process_threads": threads,
    }


_runtime_lock = threading.Lock()
_known_models = set()


def update_ollama_runtime_metrics():
    from src.core.config import settings

    with _runtime_lock:
        try:
            url = settings.models.ollama_base_url.rstrip("/") + "/api/ps"
            with urllib.request.urlopen(url, timeout=1.5) as response:
                models = json.loads(response.read())["models"]
            current = {m["name"]: m for m in models}
            _known_models.update(current)
            for name in _known_models:
                model = current.get(name)
                OLLAMA_MODEL_LOADED.labels(model=name).set(int(model is not None))
                OLLAMA_VRAM_USAGE_BYTES.labels(model=name).set(
                    model.get("size_vram", 0) if model else 0
                )
            OLLAMA_UP.set(1)
            return {"loaded": bool(models)}
        except Exception:
            OLLAMA_UP.set(0)
            for name in _known_models:
                OLLAMA_MODEL_LOADED.labels(model=name).set(float("nan"))
                OLLAMA_VRAM_USAGE_BYTES.labels(model=name).set(float("nan"))
            return {"error": True}


_metrics_server_started = False
_resource_thread_started = False


def _background_resource_monitor():
    while True:
        try:
            update_system_resource_metrics()
            update_session_duration()
            update_ollama_runtime_metrics()
        except Exception:
            pass  # Monitoring must not stop the application.
        time.sleep(4)


def start_metrics_server(port=8001):
    global _metrics_server_started, _resource_thread_started
    from prometheus_client import start_http_server

    if not _metrics_server_started:
        start_http_server(port)
        _metrics_server_started = True
    if not _resource_thread_started:
        threading.Thread(target=_background_resource_monitor, daemon=True).start()
        _resource_thread_started = True
    return True


def get_latest_metrics():
    update_system_resource_metrics()
    update_session_duration()
    return generate_latest()
