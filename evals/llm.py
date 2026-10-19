"""Task 5 evaluation: answers of the local LLM to questions about a stream context.

Step 1 (model run):  uv run --locked --group eval python -m evals.llm
Step 2 (after manual scoring of the review CSV printed by step 1):
                     uv run --locked --group eval python -m evals.llm --review <review.csv>

Scenarios live in evals/data/llm_scenarios.json. Automatic checks cover facts that must appear,
claims that must not, refusals on trap questions, answer length and language. A person then
scores every answer on the rubric (columns score_* in the review CSV); step 2 merges those scores
into the result file.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional

import requests

from evals.common import (
    ROOT,
    PeakMemory,
    base_parser,
    environment,
    latency_stats,
    save_result,
    timed,
)

SCENARIOS = ROOT / "evals" / "data" / "llm_scenarios.json"
RUBRIC = {
    "score_correct": "0 — неверно, 1 — частично, 2 — верно",
    "score_grounded": "0 — опирается на выдумку, 1 — частично, 2 — только на контекст",
    "score_style": "0 — не по-русски или не по формату, 1 — с недочётами, 2 — по-русски, 2-4 предложения",
    "hallucination": "1 — есть утверждение, которого нет в контексте; иначе 0",
}
REFUSAL = re.compile(
    r"нет (данных|информации|сведений)|не (указан|видно|известно|упомина|могу (сказать|определить|точно))"
    r"|недостаточно|неизвестн|не располагаю|не сказано|не говорил|нельзя (сказать|определить)",
    re.IGNORECASE,
)


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s])


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / max(len(letters), 1)


def auto_checks(scenario: Dict, answer: str) -> Dict:
    """Pattern checks; they flag answers for review and never replace the manual rubric."""
    missing = [
        group
        for group in scenario.get("expect", [])
        if not any(re.search(p, answer, re.IGNORECASE) for p in group)
    ]
    forbidden = [p for p in scenario.get("forbid", []) if re.search(p, answer, re.IGNORECASE)]
    refused = bool(REFUSAL.search(answer))
    n_sent = sentence_count(answer)
    return {
        "facts_ok": not missing,
        "missing_facts": missing,
        "forbidden_hits": forbidden,
        "trap_refused": refused if scenario.get("trap") else None,
        "sentences": n_sent,
        "format_ok": 1 <= n_sent <= 4,
        "russian_ok": cyrillic_share(answer) >= 0.7,
    }


def ollama_memory(base_url: str, model: str) -> Optional[Dict]:
    try:
        models = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=3).json()["models"]
    except (requests.RequestException, KeyError, ValueError):
        return None
    for m in models:
        if m.get("name") == model or m.get("model") == model:
            return {
                "size_gb": round(m.get("size", 0) / 1024**3, 2),
                "size_vram_gb": round(m.get("size_vram", 0) / 1024**3, 2),
                "context_length": m.get("context_length"),
            }
    return None


def run(args) -> None:
    from src.core.config import settings
    from src.llm_brain import LocalLLMBrain

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.limit:
        scenarios = scenarios[: args.limit]
    brain = LocalLLMBrain()
    rows, latencies, ttft, speeds = [], [], [], []
    with PeakMemory() as memory:
        for i, sc in enumerate(scenarios):
            ctx = sc["context"]
            res, ms = timed(
                brain.answer_stream_question,
                user_question=sc["question"],
                streamer_speech=ctx.get("speech", ""),
                chat_mood=ctx.get("mood", ""),
                channel_name=ctx.get("channel", ""),
                recent_chat_messages=ctx.get("chat", []),
                scene_description=ctx.get("scene", ""),
                detected_objects=ctx.get("objects", ""),
                speech_history=ctx.get("speech_history", []),
                chat_stats=ctx.get("chat_stats"),
                vlm_description=ctx.get("vlm", ""),
            )
            if i > 0:  # The first request includes model loading; it is reported separately.
                latencies.append(ms)
                ttft.append(res.load_ms + res.prompt_eval_ms)
                speeds.append(res.speed_tps)
            checks = auto_checks(sc, res.reply)
            rows.append(
                {
                    "id": sc["id"],
                    "category": sc["category"],
                    "trap": sc.get("trap", False),
                    "question": sc["question"],
                    "answer": res.reply,
                    "latency_ms": round(ms, 1),
                    "load_ms": res.load_ms,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "speed_tps": res.speed_tps,
                    **checks,
                }
            )
            print(f"[llm] {sc['id']}: facts_ok={checks['facts_ok']} {res.reply[:80]!r}")

    traps = [r for r in rows if r["trap"]]
    normal = [r for r in rows if not r["trap"]]
    metrics = {
        "scenarios": len(rows),
        "auto": {
            "facts_ok_rate": _rate(r["facts_ok"] for r in normal),
            "trap_refusal_rate": _rate(r["trap_refused"] for r in traps),
            "forbidden_hit_rate": _rate(bool(r["forbidden_hits"]) for r in rows),
            "format_ok_rate": _rate(r["format_ok"] for r in rows),
            "russian_ok_rate": _rate(r["russian_ok"] for r in rows),
        },
        "latency_answer": latency_stats(latencies),
        "ttft_approx": latency_stats(ttft),
        "speed_tps_mean": round(sum(speeds) / len(speeds), 1) if speeds else None,
        "first_request_ms": rows[0]["latency_ms"] if rows else None,
        "ollama_memory": ollama_memory(settings.models.ollama_base_url, brain.model_name),
        "process_peak_rss_mb": round(memory.peak_mb, 1),
        "answers": rows,
    }
    payload = {
        "task": "llm",
        "environment": environment("ollama"),
        "config": {
            "model": brain.model_name,
            "ollama_url": settings.models.ollama_base_url,
            "num_ctx": settings.models.ollama_context,
            "seed": settings.models.ollama_seed,
            "temperature": 0.7,
            "num_predict": 150,
            "think": False,
            "scenarios_file": str(SCENARIOS.relative_to(ROOT)),
        },
        "metrics": metrics,
        "summary": {
            "model": brain.model_name,
            "main_metric": "Оценка по рубрике (среднее, 0–2) — после ручной проверки",
            "value": None,
            "latency_p95_ms": metrics["latency_answer"].get("p95_ms"),
            "peak_rss_mb": None,
            "speed_tps": metrics["speed_tps_mean"],
        },
    }
    result_path = save_result("llm", payload, args.output_dir)
    review_path = result_path.with_suffix(".review.csv")
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "trap", "question", "answer", "auto_flags", *RUBRIC])
        for r in rows:
            flags = []
            if not r["facts_ok"] and not r["trap"]:
                flags.append(f"нет фактов: {r['missing_facts']}")
            if r["forbidden_hits"]:
                flags.append(f"запрещённое: {r['forbidden_hits']}")
            if r["trap"] and not r["trap_refused"]:
                flags.append("ловушка без отказа")
            if not r["format_ok"]:
                flags.append(f"предложений: {r['sentences']}")
            writer.writerow(
                [
                    r["id"],
                    r["category"],
                    r["trap"],
                    r["question"],
                    r["answer"],
                    "; ".join(flags),
                    "",
                    "",
                    "",
                    "",
                ]
            )
    print(f"[llm] Score the answers in {review_path} (rubric: {RUBRIC}), then run --review.")


def _rate(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(bool(v) for v in values) / len(values), 3) if values else None


def review(args) -> None:
    review_path = args.review
    result_path = Path(str(review_path).replace(".review.csv", ".json"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    with open(review_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unscored = [r["id"] for r in rows if any(r[k].strip() == "" for k in RUBRIC)]
    if unscored:
        raise SystemExit(f"Unscored rows: {unscored}")
    scores = {k: [float(r[k]) for r in rows] for k in RUBRIC}
    traps = [r for r in rows if r["trap"] == "True"]
    manual = {k: round(sum(v) / len(v), 3) for k, v in scores.items() if k != "hallucination"} | {
        "rubric_mean": round(
            sum(sum(scores[k]) for k in RUBRIC if k != "hallucination") / (3 * len(rows)), 3
        ),
        "hallucination_rate": round(sum(scores["hallucination"]) / len(rows), 3),
        "trap_hallucination_rate": round(
            sum(float(r["hallucination"]) for r in traps) / len(traps), 3
        )
        if traps
        else None,
        "reviewed_rows": len(rows),
    }
    payload["metrics"]["manual"] = manual
    payload["summary"]["value"] = manual["rubric_mean"]
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[llm] Manual scores merged into {result_path}: {manual}")


def main(argv=None):
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, help="Scored review CSV to merge into its result")
    args = parser.parse_args(argv)
    review(args) if args.review else run(args)


if __name__ == "__main__":
    main()
