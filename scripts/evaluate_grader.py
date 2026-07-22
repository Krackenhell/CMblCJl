from __future__ import annotations

import json
from pathlib import Path

from meaning_trainer.database import get_assignment, init_database
from meaning_trainer.grading import grade_structured_answer


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "grading_benchmark.json"


def evaluate() -> dict[str, float | int]:
    init_database()
    cases = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    exact_scores = 0
    false_accepts = 0
    false_rejects = 0
    absolute_error = 0.0
    for case in cases:
        result = grade_structured_answer(
            get_assignment(int(case["assignment_id"])), str(case["answer"])
        )
        if result is None:
            raise RuntimeError(f'No deterministic grade for {case["name"]}')
        predicted = float(result["score"])
        expected = float(case["expected_score"])
        difference = abs(predicted - expected)
        absolute_error += difference
        exact_scores += difference < 1e-6
        false_accepts += predicted == 1.0 and expected < 1.0
        false_rejects += predicted < 1.0 and expected == 1.0
    total = len(cases)
    return {
        "cases": total,
        "exact_score_accuracy": exact_scores / total,
        "mean_absolute_error": absolute_error / total,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
