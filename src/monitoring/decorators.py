"""Telemetry decorators and safe dispatchers for Prometheus metrics."""

import time
import functools
import logging
from typing import Callable

logger = logging.getLogger("telemetry")


def safe_telemetry_dispatch(func: Callable, *args, **kwargs) -> None:
    """Safely executes a telemetry or metric-recording call without propagating exceptions."""
    try:
        func(*args, **kwargs)
    except Exception as exc:
        logger.debug("Telemetry dispatch ignored exception: %s", exc)


def track_model_latency(model_name: str, record_func: Callable = None):
    """Decorator to measure and record execution latency of model inference functions."""

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            duration_s = time.perf_counter() - start
            if record_func:
                safe_telemetry_dispatch(record_func, model_name, duration_s)
            return result

        return wrapper

    return decorator
