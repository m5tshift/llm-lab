"""Task 1 evaluation: emotion classification on the CEDR test split.

Usage: uv run --locked --group eval python -m evals.text [--limit N] [--threshold 0.5]

CEDR labels are joy, sadness, surprise, fear, anger; an empty label list means no emotion. The
model has the same five emotions plus an explicit no_emotion label (multi-label, sigmoid).
"""

from typing import Dict, List, Sequence

from evals.common import (
    PeakMemory,
    base_parser,
    environment,
    latency_stats,
    multilabel_report,
    classification_report,
    save_result,
    subsample,
    timed,
    warmup,
)
from src.nlp_sentiment import EMOTION_TO_SENTIMENT, ChatSentimentAnalyzer

EMOTIONS = ["joy", "sadness", "surprise", "fear", "anger"]
LABELS = ["no_emotion", *EMOTIONS]
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


def gold_labels(label_ids: Sequence[int], names: Sequence[str]) -> List[str]:
    return [names[i] for i in label_ids] or ["no_emotion"]


def predicted_labels(scores: Dict[str, float], threshold: float) -> List[str]:
    """Emotions above the threshold; no emotion when none passes (no_emotion is not combined)."""
    emotions = [e for e in EMOTIONS if scores.get(e, 0.0) >= threshold]
    return emotions or ["no_emotion"]


def top1(scores: Dict[str, float]) -> str:
    return max(LABELS, key=lambda label: scores.get(label, 0.0))


def main(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="sagteam/cedr_v1")
    parser.add_argument("--config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args(argv)

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split)
    names = ds.features["labels"].feature.names
    rows = subsample(list(ds), args.limit, args.seed)
    texts = [r["text"] for r in rows]
    gold = [gold_labels(r["labels"], names) for r in rows]
    print(f"[text] {len(texts)} sentences from {args.dataset}/{args.config}:{args.split}")

    with PeakMemory() as memory:
        analyzer, load_ms = timed(ChatSentimentAnalyzer, device=args.device)
        scores = analyzer.predict_scores(texts, batch_size=args.batch_size)

        # Latency of single messages as the application sends them, after warm-up.
        sample = texts[: args.latency_samples]
        warmup(analyzer.analyze_message, sample)
        single_ms = [timed(analyzer.analyze_message, t)[1] for t in sample]
        _, batch_ms = timed(analyzer.predict_scores, texts, args.batch_size)

    predicted = [predicted_labels(s, args.threshold) for s in scores]
    emotions = multilabel_report(gold, predicted, LABELS)
    top1_hit = sum(top1(s) in g for s, g in zip(scores, gold)) / len(gold)

    # Derived sentiment, checked only where the gold annotation has exactly one label.
    single = [(g[0], top1(s)) for g, s in zip(gold, scores) if len(g) == 1]
    sentiment = classification_report(
        [EMOTION_TO_SENTIMENT[g] for g, _ in single],
        [EMOTION_TO_SENTIMENT[p] for _, p in single],
        SENTIMENTS,
    )

    by_source = {}
    for source in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        by_source[source] = multilabel_report(
            [gold[i] for i in idx], [predicted[i] for i in idx], LABELS
        )["macro_f1"]

    single_stats = latency_stats(single_ms)
    payload = {
        "task": "text",
        "environment": environment(args.device),
        "config": {
            "model": analyzer.model_name,
            "dataset": f"{args.dataset}/{args.config}:{args.split}",
            "n": len(texts),
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "metrics": {
            "emotions_multilabel": emotions,
            "top1_in_gold": round(top1_hit, 4),
            "macro_f1_by_source": by_source,
            "derived_sentiment_single_label": sentiment | {"n": len(single)},
            "latency_single_message": single_stats,
            "throughput_msgs_per_s": round(len(texts) / (batch_ms / 1000), 1),
            "model_load_ms": round(load_ms, 1),
            "peak_rss_mb": round(memory.peak_mb, 1),
        },
        "summary": {
            "model": analyzer.model_name,
            "main_metric": "macro-F1 (6 меток, CEDR test)",
            "value": emotions["macro_f1"],
            "latency_p95_ms": single_stats.get("p95_ms"),
            "peak_rss_mb": round(memory.peak_mb, 1),
        },
    }
    save_result("text", payload, args.output_dir)
    print(
        f"[text] macro-F1={emotions['macro_f1']} micro-F1={emotions['micro_f1']} "
        f"top1={top1_hit:.3f} p95={single_stats.get('p95_ms')} ms"
    )


if __name__ == "__main__":
    main()
