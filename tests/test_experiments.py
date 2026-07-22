from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_knowledge_tracing import calibration_error, run_experiment


def test_calibration_error_is_zero_for_perfect_group_calibration():
    assert calibration_error([0, 1], [0.5, 0.5], bins=2) == 0


def test_knowledge_tracing_experiment_has_baselines_and_calibration_metrics():
    report = run_experiment(seed=42)

    assert report["dataset"]["test_events"] > 500
    assert set(report["test_metrics"]) == {
        "global_mean",
        "latest_evidence",
        "ema",
        "bkt",
        "bkt_isotonic",
    }
    for result in report["test_metrics"].values():
        assert set(result) == {"brier", "log_loss", "ece_10", "accuracy_0_5"}
        assert all(0 <= value <= 1 for value in result.values())


def test_semantic_knowledge_check_benchmark_has_pass_partial_and_fail_cases():
    path = Path(__file__).parents[1] / "data" / "knowledge_check_semantic_benchmark.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) >= 10
    assert any(case["min_score"] >= 0.75 for case in cases)
    assert any(case["min_score"] < 0.75 <= case["max_score"] for case in cases)
    assert any(case["max_score"] < 0.3 for case in cases)
