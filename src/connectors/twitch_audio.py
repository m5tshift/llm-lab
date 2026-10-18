"""Twitch Stream Audio Capture Connector via FFmpeg."""

import os
import time
import threading
import subprocess
import collections
from typing import Optional, Callable, List


class TwitchAudioListener:
    def __init__(
        self,
        stream_url_getter: Callable[[], Optional[str]],
        audio_transcriber=None,
        chunk_duration_s: int = 3,
        interval_s: int = 5,
        language: str = "russian",
    ):
        self._get_url = (
            stream_url_getter if callable(stream_url_getter) else (lambda: stream_url_getter)
        )
        self.transcriber = audio_transcriber
        self.chunk_duration_s = chunk_duration_s
        self.interval_s = interval_s
        self.language = language

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current_proc: Optional[subprocess.Popen] = None
        self._latest_speech = "Слушаем стримера..."
        self._speech_history = collections.deque(maxlen=8)
        self._latest_audio_path: Optional[str] = None
        self._last_update_time = 0.0
        self._is_active = False
        self._counter = 0

    def start(self):
        if not self.transcriber:
            self._latest_speech = "(Модель Whisper не загружена)"
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._is_active = False
        self._cleanup_files()

    def get_latest_speech(self) -> str:
        with self._lock:
            return self._latest_speech

    def get_speech_history(self) -> List[str]:
        with self._lock:
            return list(self._speech_history)

    def get_latest_audio_path(self) -> Optional[str]:
        with self._lock:
            if self._latest_audio_path and os.path.exists(self._latest_audio_path):
                return self._latest_audio_path
            return None

    def _cleanup_files(self):
        pid = os.getpid()
        for i in range(3):
            fn = f"stream_snippet_{pid}_{i}.wav"
            if os.path.exists(fn):
                try:
                    os.remove(fn)
                except Exception:
                    pass

    def _worker_loop(self):
        self._is_active = True
        pid = os.getpid()

        while not self._stop_event.is_set():
            stream_url = self._get_url()
            if not stream_url:
                time.sleep(1.0)
                continue

            target_wav = f"stream_snippet_{pid}_{self._counter % 3}.wav"
            tmp_recording = f"temp_rec_{pid}.wav"

            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    stream_url,
                    "-t",
                    str(self.chunk_duration_s),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    tmp_recording,
                ]
                with self._lock:
                    if self._stop_event.is_set():
                        break
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    self._current_proc = proc

                try:
                    proc.wait(timeout=self.chunk_duration_s + 4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                finally:
                    with self._lock:
                        self._current_proc = None

                if self._stop_event.is_set():
                    break

                if os.path.exists(tmp_recording) and os.path.getsize(tmp_recording) > 1000:
                    if os.path.exists(target_wav):
                        try:
                            os.remove(target_wav)
                        except Exception:
                            pass
                    os.replace(tmp_recording, target_wav)
                    self._counter += 1

                    abs_target = os.path.abspath(target_wav)
                    with self._lock:
                        self._latest_audio_path = abs_target

                    if self._stop_event.is_set():
                        break

                    t_inf = time.perf_counter()
                    res = self.transcriber.transcribe_audio_file(abs_target, language=self.language)
                    raw_text = (
                        res.transcription
                        if hasattr(res, "transcription")
                        else res.get("transcription", "")
                    )
                    raw_text = raw_text.strip()
                    inf_ms = (time.perf_counter() - t_inf) * 1000

                    try:
                        from src.metrics import record_model_latency

                        record_model_latency("audio_whisper", inf_ms / 1000.0)
                    except Exception:
                        pass

                    clean_test = raw_text.replace(".", "").replace(" ", "").strip()
                    if clean_test:
                        with self._lock:
                            self._latest_speech = raw_text
                            self._last_update_time = time.time()
                            if not self._speech_history or self._speech_history[-1] != raw_text:
                                self._speech_history.append(raw_text)
                    else:
                        with self._lock:
                            if time.time() - self._last_update_time > 15:
                                self._latest_speech = "(Стример молчит или играет музыка)"

            except Exception:
                from src.metrics import record_processing_error
                from src.monitoring.decorators import safe_telemetry_dispatch

                safe_telemetry_dispatch(record_processing_error, "audio_whisper")
            finally:
                if os.path.exists(tmp_recording):
                    try:
                        os.remove(tmp_recording)
                    except Exception:
                        pass

            for _ in range(int(self.interval_s * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

        self._is_active = False
