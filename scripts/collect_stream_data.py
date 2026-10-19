"""Collects unlabeled data from a live Twitch channel or a local video for manual labeling.

Frames for scene labels (task 3) or detection boxes (task 4):
  uv run --locked python -m scripts.collect_stream_data frames --channel NAME --every 15 --count 40 \
      --out evals/local/scene/frames
Clips for tracking (task 4):
  uv run --locked python -m scripts.collect_stream_data clip --channel NAME --seconds 30 \
      --out evals/local/video/clips
Noise for task 2 (a fragment without speech, cut from a recorded clip):
  uv run --locked python -m scripts.collect_stream_data noise --video CLIP.mp4 --start 12 \
      --seconds 4 --out evals/local/audio/noise

A local file can replace --channel with --video PATH. For `frames --manifest`, a CSV path,label
with empty labels is appended so that a person only fills in the label column.
"""

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2


def stream_url(channel: str, quality: str) -> str:
    from src.connectors.twitch_video import TwitchStreamSource

    source = TwitchStreamSource(channel, quality)
    if not source.connect():
        raise SystemExit(f"Channel {channel} is offline or unavailable")
    url = source.stream_url
    source.release()
    return url


def collect_frames(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = args.video or stream_url(args.channel, args.quality)
    name = Path(args.video).stem if args.video else args.channel
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved, last, index = [], -float("inf"), 0
    try:
        while len(saved) < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            # Local files are sampled by their timeline, live streams by wall clock.
            now = index / fps if args.video else time.monotonic()
            index += 1
            if now - last < args.every:
                continue
            last = now
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = out / f"{name}_{stamp}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
            print(f"[frames] {len(saved)}/{args.count} {path.name}")
    finally:
        cap.release()
    if args.manifest:
        new = not args.manifest.exists()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["path", "label", "source"])
            for path in saved:
                rel = Path(os.path.relpath(path.resolve(), args.manifest.parent.resolve()))
                writer.writerow([rel.as_posix(), "", args.channel or args.video])
        print(f"[frames] Appended {len(saved)} rows to {args.manifest}")


def collect_clip(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.video or stream_url(args.channel, args.quality)
    name = Path(args.video).stem if args.video else args.channel
    path = args.out / f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if args.video:
        cmd += ["-ss", str(args.start)]
    # Audio is kept: speech-free fragments of clips become the noise set of task 2.
    cmd += ["-i", src, "-t", str(args.seconds), "-c:v", "libx264", "-preset", "veryfast"]
    cmd += ["-c:a", "aac", str(path)]
    subprocess.run(cmd, check=True)
    print(f"[clip] Saved {path}")


def collect_noise(args) -> None:
    """Cuts a mono 16 kHz WAV fragment; a person checks by ear that it has no speech."""
    args.out.mkdir(parents=True, exist_ok=True)
    name = Path(args.video).stem if args.video else args.channel
    path = args.out / f"{name}_{args.start:g}s.wav"
    src = args.video or stream_url(args.channel, args.quality)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(args.start), "-i", src]
    cmd += ["-t", str(args.seconds), "-vn", "-ac", "1", "-ar", "16000", str(path)]
    subprocess.run(cmd, check=True)
    print(f"[noise] Saved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("frames", "clip", "noise"):
        p = sub.add_parser(name)
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--channel")
        src.add_argument("--video")
        p.add_argument("--quality", default="720p")
        p.add_argument("--out", type=Path, required=True)
    frames, clip, noise = (sub.choices[n] for n in ("frames", "clip", "noise"))
    frames.add_argument("--every", type=float, default=15.0, help="Seconds between frames")
    frames.add_argument("--count", type=int, default=40)
    frames.add_argument("--manifest", type=Path, help="CSV to append path,label rows")
    clip.add_argument("--seconds", type=int, default=30)
    clip.add_argument("--start", type=float, default=0.0, help="Offset in a local video")
    noise.add_argument("--seconds", type=float, default=4.0)
    noise.add_argument("--start", type=float, default=0.0, help="Offset, seconds")
    args = parser.parse_args(argv)
    {"frames": collect_frames, "clip": collect_clip, "noise": collect_noise}[args.command](args)


if __name__ == "__main__":
    main()
