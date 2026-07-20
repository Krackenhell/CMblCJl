from __future__ import annotations

from pathlib import Path

import pytest

from vivatrace.local_llm import LLMTrace, LocalLLM, LocalLLMError
from vivatrace.models import ProbeQuestion
from vivatrace.models import Evidence


def test_missing_local_model_blocks_assessment(tmp_path, monkeypatch):
    llm = LocalLLM()
    llm.model_path = tmp_path / "missing.gguf"
    llm.server_path = tmp_path / "missing-server.exe"
    monkeypatch.setattr(llm, "_is_running", lambda *_: False)

    with pytest.raises(LocalLLMError, match="не установлена"):
        llm.ensure_available()


def test_answer_evidence_is_explicitly_traced_to_local_llm(monkeypatch):
    llm = LocalLLM()
    trace = LLMTrace(
        trace_id="local-test-call",
        backend="llama.cpp",
        model="Qwen2.5-7B-Instruct-Q4_K_M",
        model_sha256="abc123",
        stage="оценка ответа viva",
        duration_ms=120,
        created_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        llm,
        "_call_json",
        lambda *args, **kwargs: (
            {
                "score": 0.04,
                "confidence": 0.98,
                "verdict": "incorrect",
                "what_was_correct": "Предметного ответа нет.",
                "what_needs_improvement": "Нужно объяснить связь с настоящим.",
                "correct_answer": "Действие имеет результат сейчас.",
                "typo_handling": "Опечаток нет.",
                "rationale": "Ответ не содержит предметного объяснения.",
                "misconception": "нерелевантный ответ",
            },
            trace,
        ),
    )
    assignment = {"instructions": "Explain the tense.", "rubric": {"criteria": ["rule"]}}
    question = ProbeQuestion(
        id="q1",
        skill_id="eng_present_perfect",
        text="Why is Present Perfect used?",
        purpose="Проверить понимание связи с настоящим.",
        expected_concepts=(),
    )

    evidence, returned_trace = llm.evaluate_answer(assignment, question, "вся информация")

    assert evidence.score == 0.04
    assert evidence.source == "local_llm"
    assert evidence.evaluator_model == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert evidence.trace_id == "local-test-call"
    assert evidence.question_text == "Why is Present Perfect used?"
    assert evidence.rule_id == "eng_present_perfect"
    assert evidence.correct_answer == "Действие имеет результат сейчас."
    assert returned_trace == trace


def test_identity_reports_local_files_without_api_key(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    server = tmp_path / "llama-server.exe"
    manifest = tmp_path / "manifest.json"
    model.write_bytes(b"model")
    server.write_bytes(b"server")
    manifest.write_text(
        '{"model":"Local Test Model","model_sha256":"deadbeef"}', encoding="utf-8"
    )
    llm = LocalLLM()
    llm.model_path = model
    llm.server_path = server
    llm.manifest_path = manifest
    monkeypatch.setattr(llm, "_is_running", lambda *_: False)

    identity = llm.identity()

    assert identity["ready"] is True
    assert identity["backend"] == "llama.cpp"
    assert identity["model_sha256"] == "deadbeef"
    assert Path(identity["model_path"]) == model


def test_remediation_branch_and_rule_content_are_grounded(monkeypatch):
    llm = LocalLLM()
    trace = LLMTrace(
        trace_id="local-grounding",
        backend="llama.cpp",
        model="Qwen2.5-3B-Instruct-Q4_K_M",
        model_sha256="abc123",
        stage="персональный следующий шаг",
        duration_ms=100,
        created_at="2026-01-01T00:00:00+00:00",
    )
    model_outputs = iter(
        [
            {
                "branch": "transfer",
                "student_activity": {
                    "title": "Неверная ветка модели",
                    "why": "—",
                    "explanation": "Ошибочное объяснение.",
                    "worked_example": "Ошибочный пример.",
                    "practice_task": "Ошибочная практика.",
                    "success_criteria": "—",
                },
            },
            {
                "focus_topic": "Артикли",
                "reason": "Один ответ ниже порога.",
                "lesson_plan": "Разбор; пример; практика.",
                "evidence_summary": "Не различено общее понятие.",
            },
        ]
    )
    monkeypatch.setattr(
        llm,
        "_call_json",
        lambda *args, **kwargs: (next(model_outputs), trace),
    )
    assignment = {
        "topic": "A/an, the и нулевой артикль",
        "skill_ids": ["eng_articles"],
    }
    evidence = Evidence(
        skill_id="eng_articles",
        score=0.1,
        confidence=0.95,
        quote="не знаю",
        rationale="Правило не объяснено.",
        rule_id="eng_articles",
        what_needs_improvement="Нужно отличить общее понятие от конкретного.",
        correct_answer="Travel употребляется без артикля как общее понятие.",
    )

    result, traces = llm.finalize_learning(
        assignment, {"is_correct": False}, [evidence], []
    )

    assert result["branch"] == "remediation"
    assert "Нулевой артикль" in result["student_activity"]["explanation"]
    assert "Travel can teach us a lot." in result["student_activity"]["worked_example"]
    assert len(traces) == 2
