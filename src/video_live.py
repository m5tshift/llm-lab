"""Module 4: Video Processing - Real-time Live Stream & Object Detection.

Framework: Ultralytics YOLO (PyTorch) + OpenCV
Task: Real-time Object Detection & Tracking in Video / Live Stream
Model: Ultralytics YOLO26 Small (yolo26s.pt, 9.5M params, mAP50-95: 48.6%)
"""

import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from src.core.config import settings
from src.core.schemas import VideoDetectionResult, DetectedObject
from src.core.interfaces import BaseVideoDetector


class LiveStreamVideoDetector(BaseVideoDetector):
    """Real-time object detection on live video feeds using Ultralytics YOLO."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.models.yolo_model
        print(f"[Video] Loading YOLO model: '{self.model_name}'...")
        start = time.perf_counter()
        self.model = YOLO(self.model_name)
        self.load_time = time.perf_counter() - start
        print(f"[Video] YOLO model loaded successfully in {self.load_time:.2f}s")

    def process_frame(self, frame: np.ndarray) -> VideoDetectionResult:
        """Runs object detection on a single frame."""
        start = time.perf_counter()
        results = self.model(frame, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        detected_objects: List[DetectedObject] = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            detected_objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=round(conf, 3),
                    box=[round(x, 1) for x in box.xyxy[0].tolist()],
                )
            )

        annotated_frame = results.plot()

        return VideoDetectionResult(
            annotated_frame=annotated_frame,
            detected_objects=detected_objects,
            latency_ms=round(latency_ms, 2),
        )

    def detect_frame(self, frame: np.ndarray) -> VideoDetectionResult:
        """Alias for process_frame for backwards compatibility."""
        return self.process_frame(frame)

    def run_live_feed(
        self, source=0, max_frames: Optional[int] = None, display: bool = True
    ) -> Dict[str, Any]:
        """Runs live processing from webcam (source=0) or a video file path."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        frame_count = 0
        latencies = []
        prev_time = time.time()
        overall_start = time.perf_counter()

        print(
            f"[Video] Starting live video stream processing from source: {source} (Press 'q' in window to exit)..."
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result = self.process_frame(frame)
                latencies.append(result.latency_ms)

                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time

                annotated = result.annotated_frame
                obj_count = len(result.detected_objects)
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} | Latency: {result.latency_ms:.1f}ms | Objects: {obj_count}",
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
        }
        print(f"[Video] Stream finished. Stats: {stats}")
        return stats


if __name__ == "__main__":
    detector = LiveStreamVideoDetector()
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.process_frame(synthetic_frame)
    print(f"Processed synthetic frame in {res.latency_ms} ms")
