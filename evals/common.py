"""Shared evaluation utilities: environment capture, timing, memory and result files.

Every evaluation writes one JSON file to results/<task>/<timestamp>_<commit>.json with the same
top-level keys: task, environment, config, metrics, summary. The report table is built from
`summary` (scripts/report_table.py).
"""

import argparse
import csv
import json
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOCAL_DATA_DIR = ROOT / "evals" / "local"


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit or "unknown"
    except OSError:
        return "unknown"


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment(device: str) -> Dict[str, Any]:
    """Hardware and software facts recorded next to every metric."""
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "device": device,
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "ultralytics", "datasets", "jiwer")
        },
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        if device.startswith("cuda") and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


class PeakMemory:
    """Samples the process resident set size in a background thread; reports the peak in MB."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._process.memory_info().rss / 1024**2)
            time.sleep(self.interval_s)

    def __enter__(self) -> "PeakMemory":
        self.baseline_mb = self._process.memory_info().rss / 1024**2
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._process.memory_info().rss / 1024**2)


def timed(fn: Callable, *args, **kwargs):
    """Returns (result, elapsed milliseconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def warmup(fn: Callable, inputs: Sequence, n: int = 3) -> None:
    """Runs n untimed calls so that lazy initialization is excluded from latency."""
    for item in list(inputs)[:n]:
        fn(item)


def latency_stats(values_ms: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values_ms), dtype=float)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean_ms": round(float(values.mean()), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(float(values.max()), 2),
    }


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> Dict[str, Any]:
    """Single-label accuracy, per-class precision/recall/F1, macro-F1 and confusion matrix."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    per_class = {}
    for label, i in index.items():
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        per_class[label] = _prf(tp, fp, fn) | {"support": int(matrix[i, :].sum())}
    present = [label for label in labels if per_class[label]["support"] > 0]
    return {
        "accuracy": round(float(np.trace(matrix) / max(matrix.sum(), 1)), 4),
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in present])), 4)
        if present
        else 0.0,
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": matrix.tolist()},
    }


def multilabel_report(
    y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]], labels: Sequence[str]
) -> Dict[str, Any]:
    """Multi-label per-class precision/recall/F1, macro- and micro-F1, exact-match ratio."""
    per_class = {}
    totals = np.zeros(3)
    for label in labels:
        tp = sum(label in t and label in p for t, p in zip(y_true, y_pred))
        fp = sum(label not in t and label in p for t, p in zip(y_true, y_pred))
        fn = sum(label in t and label not in p for t, p in zip(y_true, y_pred))
        totals += (tp, fp, fn)
        per_class[label] = _prf(tp, fp, fn) | {"support": int(tp + fn)}
    exact = np.mean([set(t) == set(p) for t, p in zip(y_true, y_pred)]) if y_true else 0.0
    return {
        "macro_f1": round(float(np.mean([per_class[c]["f1"] for c in labels])), 4),
        "micro_f1": _prf(*totals)["f1"],
        "exact_match": round(float(exact), 4),
        "per_class": per_class,
    }


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def read_manifest(path: Path, required: Sequence[str]) -> List[Dict[str, str]]:
    """Reads a CSV manifest; relative file paths are resolved against the manifest folder."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in required if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for row in rows:
        if "path" in row and row["path"] and not Path(row["path"]).is_absolute():
            row["path"] = str((path.parent / row["path"]).resolve())
    return rows


def save_result(task: str, payload: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
    """Writes the result JSON and returns its path."""
    folder = (output_dir or RESULTS_DIR) / task
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}_{payload['environment']['git_commit']}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] Result saved: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda for a ROCm build")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N examples")
    parser.add_argument("--seed", type=int, default=42, help="Seed for subsampling")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: results/")
    return parser


def subsample(items: List[Any], limit: Optional[int], seed: int) -> List[Any]:
    """Deterministic subsample that keeps the original order."""
    if not limit or limit >= len(items):
        return items
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(items), size=limit, replace=False).tolist())
    return [items[i] for i in keep]
