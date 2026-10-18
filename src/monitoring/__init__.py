"""Monitoring and observability package."""

from src.monitoring.decorators import safe_telemetry_dispatch, track_model_latency

__all__ = ["safe_telemetry_dispatch", "track_model_latency"]
