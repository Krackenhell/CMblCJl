from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivatrace.local_llm import (
    LLMTrace,
    LocalLLM,
    LocalLLMError,
    check_article_cloze,
    generated_question_is_valid,
    sanitize_mixed_modal_negation,
)
from vivatrace.models import ProbeQuestion
from vivatrace.models import Evidence


ARTICLE_ASSIGNMENT = {
    "subject": "Английский язык · B2",
    "topic": "A/an, the и нулевой артикль",
    "instructions": (
        "Insert a/an, the or —: ‘Yesterday I took ___ train to Brighton. "
        "___ train was crowded. We had lunch near ___ sea. ___ food was excellent. "
        "___ travel often changes how people see the world.’"
    ),
    "skill_ids": ["eng_articles"],
    "rubric": {
        "reference_answer": "a train; The train; the sea; The food; — travel.",
        "criteria": [
            "a for first mention",
            "the for repeated or situationally specific nouns",
            "zero article for travel as a general concept",
        ],
    },
}


def test_article_cloze_finds_exact_wrong_position():
    answer = (
        "Yesterday I took a train to Brighton. a train was crowded. "
        "We had lunch near the sea. the food was excellent. "
        "travel often changes how people see the world."
    )

    result = check_article_cloze(ARTICLE_ASSIGNMENT, answer)

    assert result is not None
    assert result["score"] == 0.8
    assert result["correct"] is False
    wrong = [slot for slot in result["slots"] if not slot["correct"]]
    assert len(wrong) == 1
    assert wrong[0]["position"] == 2
    assert wrong[0]["student_evidence"] == "a train"
    assert wrong[0]["expected_phrase"] == "the train"


def test_zero_article_rejects_extra_pronoun_inserted_into_blank():
    answer = (
        "Yesterday I took a train to Brighton. The train was crowded. "
        "We had lunch near the sea. The food was excellent. "
        "I travel often changes how people see the world."
    )

    result = check_article_cloze(ARTICLE_ASSIGNMENT, answer)

    assert result is not None
    wrong = [slot for slot in result["slots"] if not slot["correct"]]
    assert len(wrong) == 1
    assert wrong[0]["position"] == 5
    assert wrong[0]["actual"] == "i"
    assert wrong[0]["student_evidence"] == "I travel"
    assert wrong[0]["expected_phrase"] == "travel"


def test_objective_article_check_overrides_false_positive_from_llm(monkeypatch):
    llm = LocalLLM()
    trace = LLMTrace(
        trace_id="local-false-positive",
        backend="llama.cpp",
        model="Qwen2.5-3B-Instruct-Q4_K_M",
        model_sha256="abc123",
        stage="проверка задания",
        duration_ms=100,
        created_at="2026-01-01T00:00:00+00:00",
    )

    def fake_call(stage, *args, **kwargs):
        if stage == "проверка задания":
            return (
                {
                    "submission_score": 0.85,
                    "is_correct": True,
                    "feedback": "Ошибок нет.",
                    "mode": "viva",
                    "skill_results": [
                        {"skill_id": "eng_articles", "score": 0.85, "diagnosis": "Верно."}
                    ],
                    "criterion_results": [
                        {
                            "criterion": "articles",
                            "status": "correct",
                            "student_evidence": "a train",
                            "issue": "Ошибок нет.",
                            "correction": "a train",
                        }
                    ],
                },
                trace,
            )
        if stage == "формирование вопроса 1":
            return (
                {
                    "skill_id": "eng_articles",
                    "text": "Какой артикль нужен перед повторным train?",
                    "expected_answer": "Нужен the, потому что поезд уже упомянут.",
                },
                trace,
            )
        return (
            {
                "skill_id": "eng_articles",
                "rule_focus": "the указывает на повторно упомянутый предмет",
                "expected_answer": "I saw a film. The film was excellent.",
            },
            trace,
        )

    monkeypatch.setattr(llm, "_call_json", fake_call)
    answer = (
        "Yesterday I took a train to Brighton. a train was crowded. "
        "We had lunch near the sea. the food was excellent. "
        "travel often changes how people see the world."
    )

    result, _ = llm.assess_submission(
        ARTICLE_ASSIGNMENT, answer, {"eng_articles": "Артикли"}
    )

    assert result["submission_score"] == 0.8
    assert result["is_correct"] is False
    assert result["mode"] == "diagnostic"
    assert result["criterion_results"][1]["status"] == "incorrect"
    assert result["criterion_results"][1]["correction"] == "the train"


def test_invalid_json_is_retried_once(monkeypatch):
    llm = LocalLLM()
    monkeypatch.setattr(llm, "ensure_fast_available", lambda: None)
    monkeypatch.setattr(
        llm,
        "identity",
        lambda: {
            "model": "quality",
            "fast_model_sha256": "abc123",
            "model_sha256": "def456",
        },
    )
    contents = iter(['{"score":', '{"score": 0.9}'])
    calls = []

    class FakeResponse:
        status = 200

        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raw = {
                "id": "retry-test",
                "model": "fast",
                "choices": [{"message": {"content": self.content}}],
            }
            return json.dumps(raw).encode("utf-8")

    def fake_urlopen(*args, **kwargs):
        calls.append(1)
        return FakeResponse(next(contents))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result, _ = llm._call_json(
        "оценка ответа viva",
        "Верни JSON.",
        {"answer": "test"},
        {"type": "object"},
        100,
        fast=True,
    )

    assert result == {"score": 0.9}
    assert len(calls) == 2


def test_structured_modal_score_bypasses_llm_grade(monkeypatch):
    llm = LocalLLM()
    trace = LLMTrace(
        trace_id="modal-question",
        backend="llama.cpp",
        model="Qwen2.5-3B-Instruct-Q4_K_M",
        model_sha256="abc123",
        stage="формирование вопроса",
        duration_ms=10,
        created_at="2026-01-01T00:00:00+00:00",
    )
    assignment = {
        "subject": "Английский язык · B2",
        "topic": "Модальные глаголы предположения",
        "instructions": "Choose the correct modal: 1) ___ 2) ___ 3) ___ 4) ___",
        "skill_ids": ["eng_modals_deduction"],
        "rubric": {
            "reference_answer": "1) can't; 2) must; 3) might/could; 4) must have.",
            "criteria": ["modal deduction"],
        },
    }
    stages = []

    def fake_call(stage, *args, **kwargs):
        stages.append(stage)
        if stage == "формирование вопроса 1":
            return (
                {
                    "skill_id": "eng_modals_deduction",
                    "text": "Почему в четвёртом пункте нужна форма must have?",
                    "expected_answer": "Она выражает сильный вывод о прошлом.",
                },
                trace,
            )
        return (
            {
                "skill_id": "eng_modals_deduction",
                "rule_focus": "для вывода о прошлом используется modal plus have plus V3",
                "expected_answer": "She must have left early — это сильный вывод о прошлом.",
            },
            trace,
        )

    monkeypatch.setattr(llm, "_call_json", fake_call)

    result, _ = llm.assess_submission(
        assignment,
        "1) can't\n2) must\n3) might\n4)",
        {"eng_modals_deduction": "Модальные глаголы предположения"},
    )

    assert result["submission_score"] == 0.75
    assert result["is_correct"] is False
    assert [item["status"] for item in result["criterion_results"]] == [
        "correct",
        "correct",
        "correct",
        "incorrect",
    ]
    assert stages == ["формирование вопроса 1", "формирование вопроса 2"]


def test_broken_or_off_topic_generated_questions_are_rejected():
    assert not generated_question_is_valid(
        {
            "text": "В предложении ",
            "expected_answer": "Someone must have told him.",
        },
        "must have",
    )
    assert not generated_question_is_valid(
        {
            "rule_focus": "Модальный глагол must выражает сильный вывод",
            "expected_answer": "Must be a good day today. Это сильный вывод.",
        },
        "must have",
        transfer=True,
    )
    assert generated_question_is_valid(
        {
            "text": "Почему для вывода о прошлом нужна форма must have?",
            "expected_answer": "Must have + V3 выражает сильный вывод о прошлом.",
        },
        "must have",
    )


def test_mixed_russian_negation_is_never_shown_inside_english_modal():
    assert sanitize_mixed_modal_negation("Нужно написать must не, а другую форму.") == (
        "Нужно написать must not, а другую форму."
    )


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
