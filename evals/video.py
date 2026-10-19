"""Task 4 evaluation: detection quality (mAP) and tracking (distinct-object counts, FPS).

Usage:
  uv run --locked --group eval python -m evals.video \
      --det-dir evals/local/video/det --clips evals/data/video_clips.csv

--det-dir holds images/ and labels/ in YOLO format, as exported by CVAT ("YOLO 1.1") or Label Studio
("YOLO"). If the export has its own class list (obj.names or classes.txt next to labels/), class ids
are remapped to the model's COCO ids by name; otherwise ids must already be COCO ids. --clips is a CSV with columns
path,class,true_count: the number of distinct objects of that class that appear in the clip,
counted by a person. Either part can be run alone.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2

from evals.common import (
    PeakMemory,
    base_parser,
    environment,
    latency_stats,
    read_manifest,
    save_result,
    timed,
)
from src.video_live import TRACKER_CONFIG, LiveStreamVideoDetector


def remap_labels(det_dir: Path, model_names: Dict[int, str], out_dir: Path) -> Path:
    """Copies the dataset with label ids converted from the export's class list to model ids."""
    names_file = next(
        (det_dir / n for n in ("obj.names", "classes.txt") if (det_dir / n).exists()), None
    )
    if names_file is None:
        return det_dir
    export_names = [n.strip() for n in names_file.read_text().splitlines() if n.strip()]
    to_model = {v: k for k, v in model_names.items()}
    unknown = [n for n in export_names if n not in to_model]
    if unknown:
        raise SystemExit(f"Classes not in the model's COCO list: {unknown}")
    shutil.copytree(det_dir / "images", out_dir / "images")
    (out_dir / "labels").mkdir(parents=True)
    for label in (det_dir / "labels").glob("*.txt"):
        lines = []
        for line in label.read_text().splitlines():
            if line.strip():
                cls, *box = line.split()
                lines.append(" ".join([str(to_model[export_names[int(cls)]]), *box]))
        (out_dir / "labels" / label.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"[video] Remapped class ids from {names_file.name}: {export_names}")
    return out_dir


def evaluate_detection(detector: LiveStreamVideoDetector, det_dir: Path, device: str) -> Dict:
    import yaml

    images = det_dir / "images"
    if not images.is_dir() or not (det_dir / "labels").is_dir():
        raise SystemExit(f"{det_dir} must contain images/ and labels/")
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = remap_labels(det_dir, detector.model.names, Path(tmp) / "data")
        data_yaml = Path(tmp) / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(data_dir.resolve()),
                    "train": "images",
                    "val": "images",
                    "names": detector.model.names,
                }
            )
        )
        metrics = detector.model.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            device=device,
            plots=False,
            verbose=False,
            project=tmp,
            name="val",
        )
    box = metrics.box
    names = detector.model.names
    per_class = {
        names[int(c)]: {"ap50": round(float(box.ap50[i]), 4), "ap50_95": round(float(box.ap[i]), 4)}
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "images": len(list(images.iterdir())),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 2) for k, v in metrics.speed.items()},
    }


def _frames(path: str, stride: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield frame
            index += 1
    finally:
        cap.release()
    return fps


def evaluate_tracking(detector: LiveStreamVideoDetector, clips: List[Dict], stride: int) -> Dict:
    per_clip, track_ms, detect_ms = [], [], []
    for clip in clips:
        detector.reset_tracking()
        frames = 0
        for frame in _frames(clip["path"], stride):
            _, ms = timed(detector.track_frame, frame, False)
            track_ms.append(ms)
            _, ms = timed(detector.process_frame, frame, False)
            detect_ms.append(ms)
            frames += 1
        predicted = detector.unique_objects().get(clip["class"], 0)
        true = int(clip["true_count"])
        per_clip.append(
            {
                "clip": Path(clip["path"]).name,
                "class": clip["class"],
                "true_count": true,
                "predicted_count": predicted,
                "abs_error": abs(predicted - true),
                "frames": frames,
                "all_classes": detector.unique_objects(),
            }
        )
    errors = [c["abs_error"] for c in per_clip]
    ratios = [c["predicted_count"] / c["true_count"] for c in per_clip if c["true_count"]]
    track = latency_stats(track_ms)
    detect = latency_stats(detect_ms)
    return {
        "clips": len(per_clip),
        "frame_stride": stride,
        "count_mae": round(sum(errors) / len(errors), 3) if errors else None,
        "count_ratio_mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
        "exact_count_rate": round(sum(e == 0 for e in errors) / len(errors), 3) if errors else None,
        "latency_track": track,
        "latency_detect_only": detect,
        "fps_track": round(1000 / track["mean_ms"], 2) if track.get("mean_ms") else None,
        "fps_detect_only": round(1000 / detect["mean_ms"], 2) if detect.get("mean_ms") else None,
        "per_clip": per_clip,
    }


def main(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--det-dir", type=Path)
    parser.add_argument("--clips", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th frame of clips")
    args = parser.parse_args(argv)
    if not args.det_dir and not args.clips:
        parser.error("pass --det-dir and/or --clips")

    metrics: Dict = {}
    with PeakMemory() as memory:
        detector, load_ms = timed(LiveStreamVideoDetector, args.model, args.device)
        if args.det_dir:
            metrics["detection"] = evaluate_detection(detector, args.det_dir, args.device)
            print(f"[video] mAP50={metrics['detection']['map50']}")
        if args.clips:
            clips = read_manifest(args.clips, ["path", "class", "true_count"])
            metrics["tracking"] = evaluate_tracking(detector, clips, args.stride)
            t = metrics["tracking"]
            print(f"[video] count MAE={t['count_mae']} FPS track={t['fps_track']}")
    metrics["model_load_ms"] = round(load_ms, 1)
    metrics["peak_rss_mb"] = round(memory.peak_mb, 1)

    tracking = metrics.get("tracking", {})
    payload = {
        "task": "video",
        "environment": environment(args.device),
        "config": {
            "model": args.model,
            "tracker": TRACKER_CONFIG,
            "det_dir": str(args.det_dir) if args.det_dir else None,
            "clips": str(args.clips) if args.clips else None,
            "imgsz": 640,
        },
        "metrics": metrics,
        "summary": {
            "model": args.model,
            "main_metric": "mAP@0.5 (кадры трансляций)",
            "value": metrics.get("detection", {}).get("map50"),
            "latency_p95_ms": tracking.get("latency_track", {}).get("p95_ms"),
            "peak_rss_mb": metrics["peak_rss_mb"],
            "fps_track": tracking.get("fps_track"),
        },
    }
    save_result("video", payload, args.output_dir)


if __name__ == "__main__":
    main()
