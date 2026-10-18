"""Shared local Ollama transport for text and image requests."""

import threading
import requests
from src.metrics import OLLAMA_REQUESTS, record_llm_tokens
from src.monitoring.decorators import safe_telemetry_dispatch

_generation_lock = threading.Lock()


def generate(base_url: str, payload: dict, timeout: float) -> dict:
    """Serialize generation within this process; return a completed response."""
    acquired = False
    model = payload.get("model", "unknown")
    task = (
        "vision"
        if (
            payload.get("images")
            or any(message.get("images") for message in payload.get("messages", []))
        )
        else "text"
    )
    endpoint = "chat" if "messages" in payload else "generate"

    def record_status(status):
        OLLAMA_REQUESTS.labels(model=model, task=task, status=status).inc()

    try:
        acquired = _generation_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("Ollama is busy; generation queue timeout")
        response = requests.post(
            f"{base_url.rstrip('/')}/api/{endpoint}", json=payload, timeout=timeout
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned status code {response.status_code}: {response.text}"
            )
        data = response.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        safe_telemetry_dispatch(record_status, "success")
        safe_telemetry_dispatch(
            record_llm_tokens,
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
            data.get("eval_duration", 0),
            model=model,
            task=task,
        )
        return data
    except Exception:
        safe_telemetry_dispatch(record_status, "error")
        raise
    finally:
        if acquired:
            _generation_lock.release()
