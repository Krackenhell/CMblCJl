from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from vivatrace.database import get_assignment, init_database
from vivatrace.local_llm import LocalLLM
from vivatrace.models import ProbeQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "local_llm_eval.json"


def main() -> None:
    init_database()
    assignment = get_assignment(2)
    llm = LocalLLM()
    skill_names = {"eng_present_perfect": "Present Perfect и Past Simple"}
    correct_answer = (
        "1) have lost — the loss has a result now. "
        "2) visited — 2023 is a finished past time. "
        "3) Have you ever tried — life experience up to now. "
        "4) has not finished — yet connects the action with now."
    )
    nonsense_submission = "1) banana 2) yesterday have 3) no idea 4) random words"

    correct_result, correct_traces = llm.assess_submission(
        assignment, correct_answer, skill_names
    )
    nonsense_result, nonsense_traces = llm.assess_submission(
        assignment, nonsense_submission, skill_names
    )
    question = ProbeQuestion(
        id="regression",
        skill_id="eng_present_perfect",
        text="Почему в предложении I have lost my keys используется Present Perfect?",
        purpose="Проверить понимание связи результата с настоящим.",
        expected_concepts=(),
    )
    answer_rows = []
    for answer in ("на первой, не знаю", "вся информация", "нужно воспроизвести"):
        evidence, trace = llm.evaluate_answer(assignment, question, answer)
        answer_rows.append(
            {"answer": answer, "evidence": asdict(evidence), "trace": asdict(trace)}
        )

    routing_checks = [
        correct_result["is_correct"] is True and correct_result["mode"] == "viva",
        nonsense_result["is_correct"] is False
        and nonsense_result["mode"] == "diagnostic",
    ]
    rejected = [row["evidence"]["score"] <= 0.1 for row in answer_rows]
    report = {
        "model": llm.identity(),
        "metrics": {
            "routing_accuracy": sum(routing_checks) / len(routing_checks),
            "nonsense_rejection_rate": sum(rejected) / len(rejected),
            "max_nonsense_score": max(row["evidence"]["score"] for row in answer_rows),
            "cases": len(routing_checks) + len(answer_rows),
        },
        "cases": {
            "correct_submission": {
                "result": serialize_assessment(correct_result),
                "traces": [asdict(trace) for trace in correct_traces],
            },
            "nonsense_submission": {
                "result": serialize_assessment(nonsense_result),
                "traces": [asdict(trace) for trace in nonsense_traces],
            },
            "nonsense_viva_answers": answer_rows,
        },
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Полный отчёт: {OUTPUT_PATH}")


def serialize_assessment(result: dict) -> dict:
    return {
        **result,
        "questions": [asdict(question) for question in result["questions"]],
    }


if __name__ == "__main__":
    main()
