"""Local, repeatable API evaluations. No cloud judge or Twitch messages.

Run with python -m scripts.run_evals; see README.md for usage.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = ["run_id", "correct", "relevant", "grounded", "notes", "rubric", "answer"]


def load_cases(path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = set()
    for case in cases:
        if case["id"] in ids or case["task"] not in {"sentiment", "vision", "reaction", "combined"}:
            raise ValueError("Duplicate case id or unsupported task")
        ids.add(case["id"])
        if case["deadline_ms"] <= 0:
            raise ValueError("Deadline must be positive")
        for pattern in case.get("required", []) + case.get("forbidden", []):
            re.compile(pattern)
    if not cases:
        raise ValueError("Empty evaluation suite")
    return cases


def fixture_image(kind):
    """Deterministic synthetic fixtures, not a stream quality dataset."""
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    if kind == "shapes":
        draw.rectangle((45, 90, 280, 270), fill="blue")
        draw.ellipse((410, 120, 530, 240), fill="red")
    elif kind == "score":
        # Bundled Pillow font avoids host-font differences.
        from PIL import ImageFont

        draw.text((100, 130), "SCORE 3 : 1", fill="black", font=ImageFont.load_default(size=48))
    elif kind != "blank":
        raise ValueError(f"Unknown fixture: {kind}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class APIBackend:
    def __init__(self, url, timeout):
        if urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Evaluations require a local API URL")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def post(self, route, **kwargs):
        response = self.session.post(self.url + route, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def __call__(self, case):
        stages = {}
        visual = None
        if case["task"] in {"vision", "combined"}:
            started = time.perf_counter()
            visual = self.post(
                "/api/image/vlm",
                files={"file": ("fixture.png", fixture_image(case["image"]), "image/png")},
                data={"prompt": case["prompt"]},
            )
            stages["vision_ms"] = (time.perf_counter() - started) * 1000
            if not visual.get("top_prediction", "").strip():
                raise ValueError("Empty visual result")
            if case["task"] == "vision":
                return {"text": visual["top_prediction"], "stages": stages, "vision": visual}
        started = time.perf_counter()
        if case["task"] == "sentiment":
            result = self.post("/api/text/sentiment", json={"text": case["text"]})
            text = result["label"]
        else:
            context = dict(case["context"])
            if visual:
                context["visual_context"] = visual["top_prediction"]
            result = self.post("/api/llm/generate-reaction", json=context)
            text = result["reply"]
        stages["text_ms"] = (time.perf_counter() - started) * 1000
        if not text.strip():
            raise ValueError("Empty text result")
        return {"text": text, "stages": stages, "response": result, "vision": visual}


def checks(case, text):
    """Lexical checks are diagnostics, not a semantic quality score."""
    result = {"nonempty": bool(text.strip())}
    if "expected_label" in case:
        result["label"] = text.upper() == case["expected_label"]
    for index, pattern in enumerate(case.get("required", [])):
        result[f"required_{index}"] = bool(re.search(pattern, text, re.I))
    for index, pattern in enumerate(case.get("forbidden", [])):
        result[f"forbidden_{index}"] = not bool(re.search(pattern, text, re.I))
    return result


def measure(case, backend, repetition):
    started = time.perf_counter()
    row = {
        "run_id": f"{case['id']}:{repetition}",
        "case_id": case["id"],
        "task": case["task"],
        "repetition": repetition,
        "deadline_ms": case["deadline_ms"],
        "rubric": case["rubric"],
        "input": case,
    }
    try:
        row.update(backend(case))
        row["status"] = "ok"
        row["checks"] = checks(case, row["text"])
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", checks={})
    row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    row["on_time"] = row["status"] == "ok" and row["latency_ms"] <= case["deadline_ms"]
    row["automatic_pass"] = row["status"] == "ok" and all(row["checks"].values())
    return row


def percentile(values, quantile):
    # Nearest rank; retain N because a small-suite p95 is often simply the maximum.
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)]


def summarize(rows):
    result = {}
    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        latencies = [r["latency_ms"] for r in group if r["status"] == "ok"]
        result[task] = {
            "n": len(group),
            "success_n": len(latencies),
            "error_rate": sum(r["status"] != "ok" for r in group) / len(group),
            "automatic_check_pass_rate": sum(r["automatic_pass"] for r in group) / len(group),
            "on_time_rate": sum(r["on_time"] for r in group) / len(group),
            "successful_latency_p50_ms": percentile(latencies, 0.5),
            "successful_latency_p95_ms": percentile(latencies, 0.95),
        }
        if task == "sentiment":
            labels = sorted({row["input"]["expected_label"] for row in group})
            scores = []
            for label in labels:
                tp = sum(
                    row.get("text") == label and row["input"]["expected_label"] == label
                    for row in group
                )
                fp = sum(
                    row.get("text") == label and row["input"]["expected_label"] != label
                    for row in group
                )
                fn = sum(
                    row.get("text") != label and row["input"]["expected_label"] == label
                    for row in group
                )
                scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            result[task]["macro_f1"] = sum(scores) / len(scores)
            result[task]["unique_cases"] = len({row["case_id"] for row in group})
    return result


def score_review(rows, review_path):
    expected = {r["run_id"]: r for r in rows if r["task"] != "sentiment"}
    reviewed = {}
    with Path(review_path).open(encoding="utf-8-sig", newline="") as file:
        for item in csv.DictReader(file):
            key = item["run_id"]
            if key not in expected or key in reviewed:
                raise ValueError(f"Unknown or duplicate review id: {key}")
            values = [item[name].strip() for name in ("correct", "relevant", "grounded")]
            if any(value not in {"", "0", "1"} for value in values):
                raise ValueError("Review scores must be 0, 1, or empty")
            reviewed[key] = None if "" in values else all(value == "1" for value in values)
    completed = {key: value for key, value in reviewed.items() if value is not None}
    good_timely = sum(value and expected[key]["on_time"] for key, value in completed.items())
    good = sum(value and expected[key]["status"] == "ok" for key, value in completed.items())
    return {
        "eligible_n": len(expected),
        "reviewed_n": len(completed),
        "quality_pass_rate_on_reviewed": good / len(completed) if completed else None,
        "correct_and_timely_rate_on_reviewed": good_timely / len(completed) if completed else None,
        "complete": len(completed) == len(expected),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "evals/cases.json")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--review", type=Path, help="Existing run directory; score its review.csv")
    args = parser.parse_args(argv)
    if args.review:
        rows = [
            json.loads(line)
            for line in (args.review / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        result = score_review(rows, args.review / "review.csv")
        write_json(args.review / "review_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    try:
        cases = load_cases(args.suite)
        # Validate all generated assets before any model requests.
        for case in cases:
            if "image" in case:
                fixture_image(case["image"])
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(f"Validated {len(cases)} synthetic cases; no model requests")
        return 0
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    backend = APIBackend(args.api_url, args.timeout)
    ollama = APIBackend(args.ollama_url, args.timeout)
    # Do not benchmark connection refusals while the server is still starting.
    try:
        readiness = backend.session.get(backend.url + "/", timeout=5)
        readiness.raise_for_status()
    except requests.RequestException as exc:
        parser.exit(2, f"API is not ready: {exc}\n")
    timestamp = datetime.now(timezone.utc)
    out = args.output or ROOT / "output/evals" / timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "timestamp_utc": timestamp.isoformat(),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "scope": "Synthetic API component and sequential vision-to-reaction eval; not live replay or ASR/YOLO evaluation",
        "repeats": args.repeats,
        "api_url": args.api_url,
        "timeout_seconds": args.timeout,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src").rglob("*.py"))
        },
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
        ),
    }
    for name, route in (("ollama_version", "/api/version"), ("ollama_models", "/api/tags")):
        try:
            response = ollama.session.get(ollama.url + route, timeout=5)
            response.raise_for_status()
            metadata[name] = response.json()
        except requests.RequestException as exc:
            metadata[name] = {"error": str(exc)}
    write_json(out / "metadata.json", metadata)
    write_json(out / "cases.json", cases)
    for kind in sorted({c["image"] for c in cases if "image" in c}):
        (out / f"{kind}.png").write_bytes(fixture_image(kind))
    # First request of each workload is recorded separately. It may already be warm.
    warmups = []
    for task in sorted({c["task"] for c in cases}):
        warmups.append(measure(next(c for c in cases if c["task"] == task), backend, 0))
    write_json(out / "warmup.json", warmups)
    rows = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as file:
        for repetition in range(1, args.repeats + 1):
            for case in cases:
                row = measure(case, backend, repetition)
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(f"{row['run_id']} {row['status']} {row['latency_ms']:.0f} ms", flush=True)
    write_json(out / "summary.json", summarize(rows))
    with (out / "review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "run_id": row["run_id"],
                "rubric": row["rubric"],
                "answer": row.get("text", row.get("error", "")),
            }
            for row in rows
            if row["task"] != "sentiment"
        )
    print(f"Results: {out}")
    return 0 if all(r["automatic_pass"] and r["on_time"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
