"""Builds the section 10 summary table of REPORT.md from the latest result of each task.

Usage: uv run --locked python -m scripts.summary_table
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = [
    ("text", "Текст"),
    ("audio", "Аудио"),
    ("image", "Изображение"),
    ("video", "Видео"),
    ("llm", "LLM"),
]


def latest(results_dir: Path, task: str):
    files = sorted((results_dir / task).glob("*.json"))
    return (json.loads(files[-1].read_text(encoding="utf-8")), files[-1]) if files else (None, None)


def fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    lines = [
        "| Задача | Модель | Главная ML-метрика | Результат | Латентность p95, мс | Память, МБ | Файл |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task, title in TASKS:
        result, path = latest(args.results_dir, task)
        if not result:
            lines.append(f"| {title} | — | нет результатов | — | — | — | — |")
            continue
        s = result["summary"]
        memory = s.get("peak_rss_mb")
        if task == "llm":
            mem = (result["metrics"].get("ollama_memory") or {}).get("size_gb")
            memory = f"{mem} ГБ (Ollama)" if mem else None
        lines.append(
            f"| {title} | `{s['model']}` | {s['main_metric']} | {fmt(s['value'])} | "
            f"{fmt(s.get('latency_p95_ms'), 1)} | {fmt(memory, 0)} | `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}` |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
