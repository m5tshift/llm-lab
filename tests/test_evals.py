"""Metric code of the evaluation scripts, checked on hand-computed examples."""

from pathlib import Path

import pytest

from evals.audio import normalize
from evals.common import classification_report, latency_stats, multilabel_report, subsample
from evals.llm import auto_checks, sentence_count
from evals.text import gold_labels, predicted_labels


def test_single_label_report_matches_hand_count():
    report = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert report["per_class"]["b"]["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert report["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1], [0, 2]]


def test_macro_f1_ignores_classes_without_support():
    report = classification_report(["a", "a"], ["a", "a"], ["a", "b"])
    assert report["macro_f1"] == 1.0


def test_multilabel_report_counts_each_label_independently():
    gold = [["joy"], ["joy", "surprise"], ["no_emotion"]]
    pred = [["joy"], ["joy"], ["anger"]]
    report = multilabel_report(gold, pred, ["no_emotion", "joy", "surprise", "anger"])
    assert report["per_class"]["joy"]["f1"] == 1.0
    assert report["per_class"]["surprise"]["recall"] == 0.0
    assert report["exact_match"] == pytest.approx(1 / 3, abs=1e-4)


def test_cedr_labels_map_to_model_labels():
    names = ["joy", "sadness", "surprise", "fear", "anger"]
    assert gold_labels([], names) == ["no_emotion"]
    assert gold_labels([0, 4], names) == ["joy", "anger"]
    assert predicted_labels({"joy": 0.7, "anger": 0.2, "no_emotion": 0.9}, 0.5) == ["joy"]
    assert predicted_labels({"joy": 0.1, "no_emotion": 0.3}, 0.5) == ["no_emotion"]


def test_wer_normalization_ignores_case_punctuation_and_yo():
    assert normalize("Ещё раз, ПРИВЕТ!  Всем…") == "еще раз привет всем"


def test_latency_stats_and_deterministic_subsample():
    stats = latency_stats([10, 20, 30, 40])
    assert stats["p50_ms"] == 25.0 and stats["n"] == 4
    items = list(range(100))
    assert subsample(items, 10, 1) == subsample(items, 10, 1)
    assert subsample(items, None, 1) == items


def test_llm_auto_checks_flag_missing_facts_and_trap_refusal():
    scenario = {"expect": [["2:1"]], "forbid": [r"\d+ лет"], "trap": False}
    assert auto_checks(scenario, "На табло 2:1 в пользу хозяев.")["facts_ok"]
    assert not auto_checks(scenario, "Счёт не виден.")["facts_ok"]
    trap = {"expect": [], "forbid": [], "trap": True}
    assert auto_checks(trap, "В контексте нет данных об этом.")["trap_refused"]
    assert not auto_checks(trap, "Стримеру 25 лет.")["trap_refused"]
    assert sentence_count("Первое. Второе! Третье?") == 3


def test_llm_eval_run_and_review_round_trip(tmp_path, monkeypatch):
    import csv
    import json
    from argparse import Namespace

    import evals.llm as llm_eval
    from src.core.schemas import LLMReplyResult

    class FakeBrain:
        model_name = "fake:1b"

        def answer_stream_question(self, user_question, **_):
            return LLMReplyResult(
                reply="Данных об этом нет. Счёт 2:1.", engine="fake", latency_ms=5.0, speed_tps=10.0
            )

    monkeypatch.setattr("src.llm_brain.LocalLLMBrain", FakeBrain)
    monkeypatch.setattr(llm_eval, "ollama_memory", lambda *_: None)
    llm_eval.run(Namespace(limit=3, output_dir=tmp_path))

    review = next(tmp_path.glob("llm/*.review.csv"))
    with open(review, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        row.update(score_correct="2", score_grounded="1", score_style="2", hallucination="0")
    with open(review, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    llm_eval.review(Namespace(review=review))
    result = json.loads(next(tmp_path.glob("llm/*.json")).read_text(encoding="utf-8"))
    assert result["metrics"]["manual"]["rubric_mean"] == pytest.approx(5 / 3, abs=1e-3)  # 0–2 scale
    assert result["summary"]["value"] == result["metrics"]["manual"]["rubric_mean"]


def test_frame_manifest_paths_resolve_from_another_folder(tmp_path):
    import cv2
    import numpy as np

    from evals.common import read_manifest
    from scripts.collect_stream_data import main as collect

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    manifest = tmp_path / "data" / "scene_labels.csv"
    frames = tmp_path / "local" / "frames"

    collect(
        [
            "frames",
            "--video",
            str(video),
            "--every",
            "0",
            "--count",
            "2",
            "--out",
            str(frames),
            "--manifest",
            str(manifest),
        ]
    )

    rows = read_manifest(manifest, ["path", "label"])
    assert len(rows) == 2
    assert all(Path(r["path"]).parent == frames.resolve() for r in rows)
