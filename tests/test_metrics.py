"""Regression checks for misleading counts and stale operational telemetry."""

import json
import math
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from src import metrics
from src.connectors.twitch_chat import TwitchChatListener
from src.ollama_client import generate


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0


def test_reading_and_reanalyzing_chat_does_not_inflate_received_count():
    listener = TwitchChatListener("test")
    before = value("twitch_chat_received_messages_total")
    # Identical text in two genuinely received messages must still count twice.
    listener._parse_privmsg(":viewer!viewer@host PRIVMSG #test :hello")
    listener._parse_privmsg(":viewer!viewer@host PRIVMSG #test :hello")
    for _ in range(5):
        assert listener.get_recent_messages() == ["hello", "hello"]
        metrics.record_chat_stats(1, 0)
    assert value("twitch_chat_received_messages_total") - before == 2


def test_frame_rate_includes_waits_and_drops_old_frames():
    now = [0.0]
    tracker = metrics.FrameRateTracker(window_seconds=5, clock=lambda: now[0])
    now[0] = 1
    assert tracker.tick() == 1
    now[0] = 2
    assert tracker.tick() == 1
    now[0] = 10  # Slow UI/network must reduce FPS despite fast detector inference.
    assert tracker.tick() == pytest.approx(0.2)


@pytest.mark.parametrize("task,images", [("text", []), ("vision", ["encoded-image"])])
def test_shared_transport_counts_tokens_once_for_each_task(monkeypatch, task, images):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "response": "ok",
        "prompt_eval_count": 10,
        "eval_count": 4,
        "eval_duration": 2_000_000_000,
    }
    monkeypatch.setattr("src.ollama_client.requests.post", lambda *a, **k: response)
    labels = {"model": "test-model", "task": task}
    before = value("twitch_llm_tokens_total", **labels, direction="output")
    generate("http://unused", {"model": "test-model", "images": images}, 1)
    assert value("twitch_llm_tokens_total", **labels, direction="output") - before == 4
    assert value("twitch_llm_generation_speed_tokens_per_second", **labels) == 2


def test_failed_generation_counts_error_without_tokens(monkeypatch):
    response = MagicMock(status_code=500, text="failure")
    monkeypatch.setattr("src.ollama_client.requests.post", lambda *a, **k: response)
    labels = {"model": "failed-model", "task": "vision", "status": "error"}
    before = value("twitch_ollama_requests_total", **labels)
    with pytest.raises(RuntimeError):
        generate("http://unused", {"model": "failed-model", "images": ["image"]}, 1)
    assert value("twitch_ollama_requests_total", **labels) - before == 1
    assert (
        value("twitch_llm_tokens_total", model="failed-model", task="vision", direction="input")
        == 0
    )


def test_ollama_unload_and_failure_do_not_leave_stale_vram(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(
        {"models": [{"name": "runtime-test", "size_vram": 1024}]}
    ).encode()
    monkeypatch.setattr(metrics.urllib.request, "urlopen", lambda *a, **k: response)
    monkeypatch.setattr(metrics, "_known_models", set())
    metrics.update_ollama_runtime_metrics()
    assert value("twitch_ollama_vram_usage_bytes", model="runtime-test") == 1024
    response.read.return_value = b'{"models": []}'
    metrics.update_ollama_runtime_metrics()
    assert value("twitch_ollama_model_loaded", model="runtime-test") == 0
    assert value("twitch_ollama_vram_usage_bytes", model="runtime-test") == 0
    response.read.side_effect = OSError("offline")
    metrics.update_ollama_runtime_metrics()
    assert value("twitch_ollama_up") == 0
    assert math.isnan(value("twitch_ollama_vram_usage_bytes", model="runtime-test"))


def test_session_stop_resets_ephemeral_values(monkeypatch):
    monkeypatch.setattr(metrics, "_session_start_time", None)
    metrics.set_stream_active(True)
    metrics.record_chat_stats(1, 0)
    metrics.set_stream_active(False)
    assert value("twitch_stream_active") == 0
    assert value("twitch_session_duration_seconds") == 0
    assert math.isnan(value("twitch_chat_positive_ratio"))


def test_export_contains_no_retired_quality_or_upgrade_metrics():
    names = {family.name for family in REGISTRY.collect()}
    assert not any(name.startswith("twitch_upgrade_") for name in names)
    assert "twitch_model_confidence_score" not in names
    assert "twitch_cycle_latest_latency_ms" not in names
    assert "twitch_audio_transcribed_chars" not in names


def test_dashboard_queries_have_no_fixed_quality_or_zero_fallbacks():
    from scripts.build_3_dashboards import dashboards

    for _, _, _, panels in dashboards():
        for panel in panels:
            for target in panel["targets"]:
                assert "vector(" not in target["expr"]
                assert "twitch_upgrade_" not in target["expr"]


def test_api_bounds_unknown_routes_and_counts_unhandled_failures(monkeypatch):
    from fastapi.testclient import TestClient
    from src import api

    client = TestClient(api.app, raise_server_exceptions=False)
    labels = {"endpoint": "unmatched", "method": "GET", "status_code": "404"}
    before = value("twitch_api_requests_total", **labels)
    assert client.get("/unknown-1").status_code == 404
    assert client.get("/unknown-2").status_code == 404
    assert value("twitch_api_requests_total", **labels) - before == 2
    broken = MagicMock()
    broken.analyze_message.side_effect = RuntimeError("model failed")
    monkeypatch.setattr(api, "nlp_model", broken)
    labels = {"endpoint": "/api/text/sentiment", "method": "POST", "status_code": "500"}
    before = value("twitch_api_requests_total", **labels)
    assert client.post("/api/text/sentiment", json={"text": "hello"}).status_code == 500
    assert value("twitch_api_requests_total", **labels) - before == 1


def test_vlm_freshness_resets_on_close_and_empty_results_are_errors():
    from src.live_vision import BackgroundVLMAnalyzer
    from types import SimpleNamespace

    model = MagicMock()
    model.classify_image.return_value = SimpleNamespace(
        top_prediction="scene", inference_latency_ms=3
    )
    analyzer = BackgroundVLMAnalyzer(model_factory=lambda: model)
    analyzer._frame_timestamp = 123
    analyzer._analyze(None)
    assert value("twitch_vlm_frame_timestamp_seconds") == 123
    before = value("twitch_processing_errors_total", component="image_vlm")
    model.classify_image.return_value.top_prediction = ""
    analyzer._analyze(None)
    assert value("twitch_processing_errors_total", component="image_vlm") - before == 1
    assert value("twitch_vlm_frame_timestamp_seconds") == 123
    analyzer.close()
    assert math.isnan(value("twitch_vlm_frame_timestamp_seconds"))
