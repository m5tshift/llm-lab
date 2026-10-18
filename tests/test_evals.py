"""Evaluation accounting must not disguise failures or unreviewed outputs."""

import csv

import pytest

from scripts.run_evals import (
    APIBackend,
    ROOT,
    checks,
    fixture_image,
    load_cases,
    measure,
    percentile,
    score_review,
    summarize,
)


def case():
    return {"id": "test", "task": "reaction", "deadline_ms": 1000, "rubric": "facts"}


def test_failure_is_not_fast_success():
    def fail(_):
        raise TimeoutError("model timeout")

    row = measure(case(), fail, 1)
    summary = summarize([row])["reaction"]
    assert not row["on_time"] and not row["automatic_pass"]
    assert summary["error_rate"] == 1
    assert summary["successful_latency_p95_ms"] is None


def test_checks_do_not_override_manual_quality():
    row = measure(case(), lambda _: {"text": "invented score"}, 1)
    assert row["automatic_pass"]
    assert "quality_pass" not in row
    assert not all(checks({"forbidden": ["score"]}, row["text"]).values())


def test_review_requires_complete_scores_and_counts_errors(tmp_path):
    rows = [measure(case(), lambda _: {"text": "answer"}, n) for n in (1, 2)]
    path = tmp_path / "review.csv"
    path.write_text("run_id,correct,relevant,grounded\ntest:1,1,1,1\ntest:2,,,\n")
    result = score_review(rows, path)
    assert result["reviewed_n"] == 1 and not result["complete"]
    rows[0]["status"] = "error"
    rows[0]["on_time"] = False
    result = score_review(rows, path)
    assert result["quality_pass_rate_on_reviewed"] == 0
    assert result["correct_and_timely_rate_on_reviewed"] == 0


@pytest.mark.parametrize("entries", ["test:1,2,1,1", "unknown,1,1,1", "test:1,1,1,1\ntest:1,1,1,1"])
def test_invalid_review_rejected(tmp_path, entries):
    path = tmp_path / "review.csv"
    path.write_text("run_id,correct,relevant,grounded\n" + entries + "\n")
    with pytest.raises(ValueError):
        score_review([measure(case(), lambda _: {"text": "ok"}, 1)], path)


def test_percentiles_and_suite():
    assert percentile([1, 2, 100], 0.95) == 100
    assert percentile([], 0.5) is None
    cases = load_cases(ROOT / "evals/cases.json")
    assert {c["task"] for c in cases} == {"sentiment", "vision", "reaction", "combined"}
    assert fixture_image("score") == fixture_image("score")


def test_local_only():
    with pytest.raises(ValueError):
        APIBackend("https://example.com", 1)


def test_sentiment_f1_counts_failed_requests():
    positive = dict(case(), task="sentiment", expected_label="POSITIVE")
    negative = dict(case(), id="neg", task="sentiment", expected_label="NEGATIVE")

    def fail(_):
        raise TimeoutError("timeout")

    rows = [measure(positive, lambda _: {"text": "POSITIVE"}, 1), measure(negative, fail, 1)]
    summary = summarize(rows)["sentiment"]
    assert summary["macro_f1"] == 0.5
    assert summary["unique_cases"] == 2


def test_generated_csv_compatible(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "correct", "relevant", "grounded"])
        writer.writeheader()
        writer.writerow({"run_id": "test:1"})
    result = score_review([measure(case(), lambda _: {"text": "ok"}, 1)], path)
    assert result["quality_pass_rate_on_reviewed"] is None
