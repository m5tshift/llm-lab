"""Module 4: Video Processing - Real-time Live Stream & Object Detection.

Framework: Ultralytics YOLO (PyTorch) + OpenCV
Task: Object detection in video frames and multi-object tracking (ByteTrack) across frames
Default model: Ultralytics YOLO26 Small (yolo26s.pt)
"""

import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set
import cv2
import numpy as np

# Dependencies come from uv.lock: forbid Ultralytics from pip-installing packages at runtime.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO  # noqa: E402

from src.core.config import settings
from src.core.schemas import VideoDetectionResult, DetectedObject
from src.core.interfaces import BaseVideoDetector


TRACKER_CONFIG = "bytetrack.yaml"


class LiveStreamVideoDetector(BaseVideoDetector):
    """Object detection on single frames and ByteTrack tracking on frame sequences."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.models.yolo_model
        self.device = device
        print(f"[Video] Loading YOLO model: '{self.model_name}'...")
        start = time.perf_counter()
        self.model = YOLO(self.model_name)
        self.load_time = time.perf_counter() - start
        print(f"[Video] YOLO model loaded successfully in {self.load_time:.2f}s")
        self._seen_tracks: Dict[str, Set[int]] = defaultdict(set)
        # track() registers tracker callbacks on its model, so tracking gets its own instance and
        # plain detection calls cannot disturb (or be disturbed by) the tracker state.
        self._track_model: Optional[YOLO] = None

    def _to_result(self, results, latency_ms: float, annotate: bool) -> VideoDetectionResult:
        detected_objects: List[DetectedObject] = []
        track_ids = results.boxes.id
        for i, box in enumerate(results.boxes):
            class_name = self.model.names[int(box.cls[0].item())]
            track_id = int(track_ids[i].item()) if track_ids is not None else None
            if track_id is not None:
                self._seen_tracks[class_name].add(track_id)
            detected_objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=round(float(box.conf[0].item()), 3),
                    box=[round(x, 1) for x in box.xyxy[0].tolist()],
                    track_id=track_id,
                )
            )
        return VideoDetectionResult(
            annotated_frame=results.plot() if annotate else None,
            detected_objects=detected_objects,
            latency_ms=round(latency_ms, 2),
            unique_objects=self.unique_objects(),
        )

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> VideoDetectionResult:
        """Detects objects on one independent frame (no identity across frames)."""
        start = time.perf_counter()
        results = self.model(frame, verbose=False, device=self.device)[0]
        latency_ms = (time.perf_counter() - start) * 1000
        return self._to_result(results, latency_ms, annotate)

    def track_frame(self, frame: np.ndarray, annotate: bool = True) -> VideoDetectionResult:
        """Detects objects and links them to tracks from previous frames of the same sequence."""
        start = time.perf_counter()
        if self._track_model is None:
            self._track_model = YOLO(self.model_name)
        results = self._track_model.track(
            frame, persist=True, tracker=TRACKER_CONFIG, verbose=False, device=self.device
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000
        return self._to_result(results, latency_ms, annotate)

    def unique_objects(self) -> Dict[str, int]:
        """Number of distinct track ids per class since the last reset."""
        return {name: len(ids) for name, ids in sorted(self._seen_tracks.items())}

    def reset_tracking(self) -> None:
        """Starts a new sequence: forgets tracker state and the distinct-object counts."""
        self._seen_tracks.clear()
        predictor = getattr(self._track_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def run_live_feed(
        self, source=0, max_frames: Optional[int] = None, display: bool = True
    ) -> Dict[str, Any]:
        """Runs live tracking from webcam (source=0) or a video file path."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        frame_count = 0
        latencies = []
        prev_time = time.time()
        overall_start = time.perf_counter()
        self.reset_tracking()

        print(
            f"[Video] Starting live video stream processing from source: {source} (Press 'q' in window to exit)..."
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result = self.track_frame(frame)
                latencies.append(result.latency_ms)

                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time

                annotated = result.annotated_frame
                obj_count = sum(result.unique_objects.values())
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} | Latency: {result.latency_ms:.1f}ms | Unique: {obj_count}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if display:
                    cv2.imshow("Twitch Bot - Real-Time Video Stream Analysis", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[Video] Stopped by user (pressed 'q').")
                        break

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        total_time_s = time.perf_counter() - overall_start
        avg_fps = frame_count / total_time_s if total_time_s > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        stats = {
            "total_frames_processed": frame_count,
            "total_time_seconds": round(total_time_s, 2),
            "average_fps": round(avg_fps, 2),
            "average_inference_latency_ms": round(avg_latency, 2),
            "unique_objects": self.unique_objects(),
        }
        print(f"[Video] Stream finished. Stats: {stats}")
        return stats


if __name__ == "__main__":
    detector = LiveStreamVideoDetector()
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.process_frame(synthetic_frame)
    print(f"Processed synthetic frame in {res.latency_ms} ms")
