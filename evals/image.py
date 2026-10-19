"""Task 3 evaluation: zero-shot scene classification on labeled stream frames.

Usage: uv run --locked --group eval python -m evals.image --manifest evals/data/scene_labels.csv

The manifest is a CSV with columns path,label (label is one of the SCENE_PROMPTS keys); rows with
an empty label are skipped. Two prompt configurations are compared on the same frames: the
ensemble of all prompts per class (used by the application) and only the first prompt per class.
"""

from pathlib import Path
from typing import Dict, List

from evals.common import (
    PeakMemory,
    base_parser,
    classification_report,
    environment,
    latency_stats,
    read_manifest,
    save_result,
    subsample,
    timed,
    warmup,
)
from src.scene_classify import SCENE_PROMPTS, SceneClassifier


def evaluate(model_name: str, prompts: Dict[str, List[str]], rows: List[Dict], device: str) -> Dict:
    labels = list(prompts)
    with PeakMemory() as memory:
        classifier, load_ms = timed(SceneClassifier, model_name, device, prompts)
        paths = [r["path"] for r in rows]
        warmup(classifier.classify_image, paths)
        preds, latencies, errors = [], [], []
        for row in rows:
            res, ms = timed(classifier.classify_image, row["path"])
            preds.append(res.top_prediction)
            latencies.append(ms)
            if res.top_prediction != row["label"]:
                errors.append(
                    {
                        "path": Path(row["path"]).name,
                        "true": row["label"],
                        "pred": res.top_prediction,
                        "confidence": res.confidence,
                    }
                )
    report = classification_report([r["label"] for r in rows], preds, labels)
    return report | {
        "latency": latency_stats(latencies),
        "model_load_ms": round(load_ms, 1),
        "peak_rss_mb": round(memory.peak_mb, 1),
        "errors": errors[:30],
    }


def main(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--compare-model", default=None, help="Optional second CLIP checkpoint on the same frames"
    )
    args = parser.parse_args(argv)

    rows = [r for r in read_manifest(args.manifest, ["path", "label"]) if r["label"].strip()]
    unknown = sorted({r["label"] for r in rows} - set(SCENE_PROMPTS))
    if unknown:
        raise SystemExit(f"Unknown labels in manifest: {unknown}; allowed: {list(SCENE_PROMPTS)}")
    rows = subsample(rows, args.limit, args.seed)
    support = {label: sum(r["label"] == label for r in rows) for label in SCENE_PROMPTS}
    print(f"[image] {len(rows)} labeled frames: {support}")

    single_prompt = {label: prompts[:1] for label, prompts in SCENE_PROMPTS.items()}
    runs = {"ensemble": evaluate(args.model, SCENE_PROMPTS, rows, args.device)}
    runs["single_prompt"] = evaluate(args.model, single_prompt, rows, args.device)
    if args.compare_model:
        runs[f"ensemble:{args.compare_model}"] = evaluate(
            args.compare_model, SCENE_PROMPTS, rows, args.device
        )
    for name, run in runs.items():
        print(f"[image] {name}: accuracy={run['accuracy']} macro-F1={run['macro_f1']}")

    main_run = runs["ensemble"]
    payload = {
        "task": "image",
        "environment": environment(args.device),
        "config": {
            "model": args.model,
            "compare_model": args.compare_model,
            "manifest": str(args.manifest),
            "n": len(rows),
            "support": support,
            "prompts": SCENE_PROMPTS,
            "seed": args.seed,
        },
        "metrics": runs,
        "summary": {
            "model": args.model,
            "main_metric": "macro-F1 (5 классов сцены, собственный набор)",
            "value": main_run["macro_f1"],
            "latency_p95_ms": main_run["latency"].get("p95_ms"),
            "peak_rss_mb": main_run["peak_rss_mb"],
        },
    }
    save_result("image", payload, args.output_dir)


if __name__ == "__main__":
    main()
