from __future__ import annotations

import argparse
import json
from pathlib import Path

from meaning_trainer.local_llm import LocalLLM
from meaning_trainer.models import ProbeQuestion


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "knowledge_check_semantic_benchmark.json"


def distance_to_band(score: float, minimum: float, maximum: float) -> float:
    if score < minimum:
        return minimum - score
    if score > maximum:
        return score - maximum
    return 0.0


def expected_label(case: dict) -> int | None:
    if float(case["min_score"]) >= 0.75:
        return 1
    if float(case["max_score"]) < 0.75:
        return 0
    return None


def evaluate(limit: int | None = None, case_id: str | None = None) -> dict:
    cases = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if case_id:
        cases = [case for case in cases if case["id"] == case_id]
    if limit:
        cases = cases[:limit]
    llm = LocalLLM()
    results = []
    for case in cases:
        question = ProbeQuestion(
            id=case["id"],
            skill_id=case["rule_id"],
            text=case["question"],
            purpose="Смысловая проверка ответов на понимание.",
            expected_concepts=tuple(tuple(group) for group in case["expected_concepts"]),
            rule_id=case["rule_id"],
            expected_answer=case["expected_answer"],
        )
        evidence, trace = llm.evaluate_answer(
            {"instructions": case["question"], "rubric": {"criteria": []}},
            question,
            case["student_answer"],
        )
        results.append(
            {
                "id": case["id"],
                "score": evidence.score,
                "min_score": case["min_score"],
                "max_score": case["max_score"],
                "within_band": case["min_score"] <= evidence.score <= case["max_score"],
                "distance_to_band": distance_to_band(
                    evidence.score, case["min_score"], case["max_score"]
                ),
                "verdict": evidence.verdict,
                "duration_ms": trace.duration_ms,
            }
        )
    classified = [
        (row, expected_label(case)) for row, case in zip(results, cases, strict=True)
        if expected_label(case) is not None
    ]
    false_accepts = sum(
        label == 0 and row["score"] >= 0.75 for row, label in classified
    )
    false_rejects = sum(
        label == 1 and row["score"] < 0.75 for row, label in classified
    )
    return {
        "cases": len(results),
        "within_band_accuracy": sum(row["within_band"] for row in results) / len(results),
        "mean_distance_to_band": sum(row["distance_to_band"] for row in results) / len(results),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "mean_latency_ms": sum(row["duration_ms"] for row in results) / len(results),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.limit, args.case)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
