"""
Twitch Video Stream Connector via Streamlink and OpenCV.
Resolves live HLS streams and provides non-blocking threaded frame capture.
"""

import time
import os
import subprocess
import threading
from typing import Optional
import cv2
import numpy as np

try:
    import streamlink
except ImportError:
    streamlink = None


class TwitchStreamSource:
    """Resolves live Twitch HLS stream via Streamlink and captures video frames / audio chunks."""

    def __init__(self, channel: str, quality: str = "720p"):
        self.channel = channel.lstrip("#").lower()
        self.quality = quality
        self.stream_url: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()

    def connect(self) -> bool:
        """Resolves live stream URL using streamlink and opens OpenCV VideoCapture."""
        if streamlink is None:
            raise RuntimeError("Streamlink is not installed. Run `uv sync --locked`.")

        twitch_url = f"https://www.twitch.tv/{self.channel}"
        print(f"[Twitch Video] Querying live streams for {twitch_url}...")

        try:
            streams = streamlink.streams(twitch_url)
        except Exception as e:
            print(f"[Twitch Video] Failed to query streams: {e}")
            return False

        if not streams:
            print(
                f"[Twitch Video] Channel '{self.channel}' is currently OFFLINE or does not exist."
            )
            return False

        # Match quality or fallback to best/worst
        available = list(streams.keys())
        print(f"[Twitch Video] Available stream qualities: {available}")

        # Match quality with prefix support (e.g. '720p' matches '720p60', '720p')
        selected_quality = None
        for key in streams.keys():
            if key.startswith(self.quality):
                selected_quality = key
                break

        if not selected_quality:
            for fallback in ["720p", "480p", "360p", "1080p", "worst", "best"]:
                for key in streams.keys():
                    if key.startswith(fallback) or key == fallback:
                        selected_quality = key
                        break
                if selected_quality:
                    break

        if not selected_quality:
            selected_quality = list(streams.keys())[0]

        self.stream_url = streams[selected_quality].url
        print(f"[Twitch Video] Connecting to stream quality '{selected_quality}'...")

        with self._cap_lock:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                print("[Twitch Video] Failed to open HLS video stream with OpenCV.")
                return False

            # Pre-buffer first frame
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame_event.set()

        # Start continuous background frame reader thread to prevent network I/O lag in main loop
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._capture_thread.start()

        print("[Twitch Video] Video stream and background frame grabber opened successfully!")
        return True

    def _capture_worker(self):
        """Continuously reads HLS frames in a background thread to prevent blocking inference."""
        while not self._stop_event.is_set():
            ret = False
            frame = None
            with self._cap_lock:
                if not self.cap or not self.cap.isOpened():
                    break
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    break

            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
            else:
                time.sleep(0.01)

        # Safely release inside worker when exited
        with self._cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def get_frame(self, wait_timeout: float = 0.3) -> Optional[np.ndarray]:
        """Returns the latest live frame immediately from memory without blocking."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        if self._new_frame_event.wait(timeout=wait_timeout):
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
        return None

    def capture_audio_chunk(
        self, duration_s: int = 5, output_wav: str = "temp_stream_audio.wav"
    ) -> Optional[str]:
        """Extracts a short audio chunk from the stream using FFmpeg for Whisper transcription."""
        if not self.stream_url:
            return None

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.stream_url,
            "-t",
            str(duration_s),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_wav,
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=duration_s + 5,
                check=True,
            )
            if os.path.exists(output_wav) and os.path.getsize(output_wav) > 1000:
                return output_wav
        except Exception:
            pass
        return None

    def stop(self):
        """Stops background capture and releases VideoCapture safely."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        with self._cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def release(self):
        """Alias for stop."""
        self.stop()
