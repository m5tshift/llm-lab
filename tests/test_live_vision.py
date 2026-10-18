"""Background VLM must never hold up the live frame loop or reuse stale context."""

import threading
import time
from types import SimpleNamespace

import numpy as np

from src.live_vision import BackgroundVLMAnalyzer


def test_background_vlm_is_nonblocking_and_uses_a_frame_copy():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    frames = []

    class Model:
        def classify_image(self, frame, **kwargs):
            frames.append(frame)
            started.set()
            release.wait(timeout=2)
            finished.set()
            return SimpleNamespace(top_prediction="Счёт 2:1", inference_latency_ms=10)

    analyzer = BackgroundVLMAnalyzer(model_factory=Model, interval_seconds=0)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert analyzer.submit(frame)
    assert started.wait(timeout=2)
    frame[:] = 255
    assert frames[0].sum() == 0
    assert not analyzer.submit(frame)
    assert analyzer.get_latest() == ""
    release.set()
    assert finished.wait(timeout=2)
    deadline = time.monotonic() + 2
    while not analyzer.get_latest() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert analyzer.get_latest() == "Счёт 2:1"
    analyzer.close()
    assert not analyzer.submit(frame)


def test_background_vlm_expires_old_result():
    done = threading.Event()

    class Model:
        def classify_image(self, frame, **kwargs):
            done.set()
            return SimpleNamespace(top_prediction="Старый кадр", inference_latency_ms=10)

    analyzer = BackgroundVLMAnalyzer(model_factory=Model, interval_seconds=20, max_age_seconds=0.01)
    assert analyzer.submit(np.zeros((2, 2, 3), dtype=np.uint8))
    assert done.wait(timeout=2)
    time.sleep(0.03)
    assert analyzer.get_latest() == ""
    analyzer.close()
