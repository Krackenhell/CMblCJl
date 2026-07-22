from __future__ import annotations

import numpy as np
import pytest

from vivatrace.grammar import offline_grammar_findings, relative_clause_evidence
from vivatrace.local_llm import normalize_voice_dialogue_result
from vivatrace.voice import (
    acoustic_fluency_metrics,
    is_assessable_spoken_turn,
    overall_speaking_score,
    voice_component_html,
)


def test_acoustic_fluency_metrics_use_audio_and_transcript() -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    voiced = 0.18 * np.sin(2 * np.pi * 220 * time_axis)
    samples = np.concatenate([voiced, np.zeros(sample_rate // 2), voiced])
    pcm16 = (samples * 32767).astype("<i2").tobytes()

    metrics = acoustic_fluency_metrics(
        pcm16,
        "I think this example is useful for our discussion today",
        sample_rate,
    )

    assert metrics["duration_seconds"] == pytest.approx(1.5, abs=0.01)
    assert metrics["signal_peak"] == pytest.approx(0.18, abs=0.01)
    assert metrics["signal_rms"] > 0.09
    assert metrics["word_count"] == 10
    assert metrics["words_per_minute"] == pytest.approx(400, abs=1)
    assert 0.25 <= metrics["pause_ratio"] <= 0.45
    assert 0 <= metrics["fluency_score"] <= 1


def test_overall_speaking_score_combines_four_dimensions() -> None:
    result = overall_speaking_score(
        {"grammar_score": 0.8, "vocabulary_score": 0.7, "relevance_score": 0.9},
        {"fluency_score": 0.6},
    )

    assert result == pytest.approx(0.75)


def test_voice_guardrail_accepts_correct_grounded_alternative() -> None:
    result, needs_fallback = normalize_voice_dialogue_result(
        {
            "reply_en": "That is interesting.",
            "feedback_ru": "В предложении нужна форма where.",
            "grammar_score": 0.85,
            "vocabulary_score": 0.8,
            "relevance_score": 0.3,
            "correction_en": "I studied in Kyoto, where attracts many visitors.",
            "next_goal_ru": "Исправить relative clause.",
        },
        transcript="I studied in Kyoto, which attracts many visitors.",
        grounded_rule={
            "summary": "Relative clauses describe people, things and places.",
            "principles": ["which can be the subject of a relative clause"],
            "examples": ["Kyoto, which attracts many visitors, is beautiful."],
        },
    )

    assert result["grammar_score"] == pytest.approx(0.85)
    assert result["relevance_score"] >= 0.8
    assert result["correction_en"] == ""
    assert "корректна" in result["feedback_ru"]
    assert result["reply_en"] == "That is interesting."
    assert needs_fallback is False


def test_voice_guardrail_normalizes_invalid_score_and_requests_quality_check() -> None:
    result, needs_fallback = normalize_voice_dialogue_result(
        {
            "reply_en": "Could you explain it?",
            "feedback_ru": "Нужна проверка.",
            "grammar_score": 0.4,
            "vocabulary_score": 3,
            "relevance_score": 0.2,
            "correction_en": "",
            "next_goal_ru": "Уточнить форму.",
        },
        transcript="I am not sure.",
        grounded_rule={"summary": "Relative clauses", "principles": [], "examples": []},
    )

    assert result["vocabulary_score"] == pytest.approx(0.3)
    assert needs_fallback is True


def test_structural_grammar_rule_distinguishes_which_from_where() -> None:
    correct = offline_grammar_findings(
        "I studied in Kyoto, which attracts many visitors.", "eng_relative_clauses"
    )
    incorrect = offline_grammar_findings(
        "I studied in Kyoto, where attracts many visitors.", "eng_relative_clauses"
    )

    assert not any(item["code"] == "RELATIVE_WHERE_MISSING_SUBJECT" for item in correct)
    finding = next(
        item for item in incorrect if item["code"] == "RELATIVE_WHERE_MISSING_SUBJECT"
    )
    assert finding["suggestions"] == [
        "I studied in Kyoto, which attracts many visitors."
    ]


def test_voice_guardrail_enforces_independent_grammar_finding() -> None:
    finding = {
        "code": "RELATIVE_WHERE_MISSING_SUBJECT",
        "message": "После where требуется подлежащее.",
        "fragment": "where attracts",
        "suggestions": ["I studied in Kyoto, which attracts many visitors."],
    }
    result, needs_fallback = normalize_voice_dialogue_result(
        {
            "reply_en": "That sounds correct. Why?",
            "feedback_ru": "Ошибок нет.",
            "grammar_score": 0.9,
            "vocabulary_score": 0.8,
            "relevance_score": 0.8,
            "correction_en": "",
            "next_goal_ru": "Продолжить.",
        },
        transcript="I studied in Kyoto, where attracts many visitors.",
        grounded_rule={"summary": "Relative clauses", "principles": [], "examples": []},
        grammar_findings=[finding],
    )

    assert result["grammar_score"] == pytest.approx(0.55)
    assert result["correction_en"] == "I studied in Kyoto, which attracts many visitors."
    assert "where attracts" in result["feedback_ru"]
    assert result["reply_en"].startswith("Almost. Try:")
    assert "Please say the corrected sentence once." in result["reply_en"]
    assert needs_fallback is False


def test_voice_turn_filter_excludes_microphone_checks_but_keeps_examples() -> None:
    assert not is_assessable_spoken_turn("Are you listening to me?")
    assert not is_assessable_spoken_turn(
        "Can you listen to one more example with a defining clause?"
    )
    assert not is_assessable_spoken_turn(
        "Yes, and check my example which I gave you. How do you like it?"
    )
    assert not is_assessable_spoken_turn(
        "But I gave you the example. Why do you ask me the second time?"
    )
    assert is_assessable_spoken_turn(
        "The woman who designed the app won an award."
    )


def test_voice_guardrail_removes_stock_follow_up_question() -> None:
    result, _ = normalize_voice_dialogue_result(
        {
            "reply_en": (
                "That is a useful example. Why did you choose that form, and how would "
                "another form change the meaning?"
            ),
            "feedback_ru": "Верный пример.",
            "grammar_score": 0.9,
            "vocabulary_score": 0.8,
            "relevance_score": 0.8,
            "correction_en": "",
            "next_goal_ru": "Продолжить.",
        },
        transcript="The woman who designed the app won an award.",
        grounded_rule={"summary": "Relative clauses", "principles": [], "examples": []},
        structural_evidence=relative_clause_evidence(
            "The woman who designed the app won an award.", "eng_relative_clauses"
        ),
    )

    assert "Why did you choose that form" not in result["reply_en"]
    assert "defining clause" in result["reply_en"]
    assert "woman" in result["reply_en"]


def test_relative_clause_audit_catches_non_human_who_and_explains_where() -> None:
    transcript = "The cafe, who serves vegan food, is very tasty."
    findings = offline_grammar_findings(transcript, "eng_relative_clauses")

    finding = next(item for item in findings if item["code"] == "RELATIVE_WHO_NON_HUMAN")
    assert "which" in finding["suggestions"][0]
    assert "where" in finding["message"]


def test_relative_clause_audit_identifies_defining_and_non_defining_examples() -> None:
    defining = relative_clause_evidence(
        "I am working on my project which I will present to ITMO University.",
        "eng_relative_clauses",
    )
    non_defining = relative_clause_evidence(
        "My laptop, which is five years old, still works well.",
        "eng_relative_clauses",
    )

    assert defining[0]["clause_type_from_transcript"] == "defining"
    assert defining[0]["antecedent"].lower().endswith("project")
    assert non_defining[0]["clause_type_from_transcript"] == "non_defining"
    assert non_defining[0]["marker_valid"] is True


def test_voice_guardrail_checks_answer_against_previous_tutor_question() -> None:
    transcript = "The woman who designs the app is very talented."
    result, _ = normalize_voice_dialogue_result(
        {
            "reply_en": "Your example is correct. Next, provide a non-defining clause.",
            "feedback_ru": "Верно.",
            "grammar_score": 0.9,
            "vocabulary_score": 0.8,
            "relevance_score": 0.9,
            "correction_en": "",
            "next_goal_ru": "Продолжить.",
        },
        transcript=transcript,
        grounded_rule={"summary": "Relative clauses", "principles": [], "examples": []},
        structural_evidence=relative_clause_evidence(transcript, "eng_relative_clauses"),
        history=[
            {
                "role": "assistant",
                "content": "Provide a non-defining clause about a person.",
            }
        ],
    )

    assert "appears defining" in result["reply_en"]
    assert result["relevance_score"] <= 0.65
    assert "предыдущий вопрос" in result["feedback_ru"]


def test_spoken_usage_audit_catches_works_good() -> None:
    findings = offline_grammar_findings(
        "My laptop, which is five years old, still works good.",
        "eng_relative_clauses",
    )

    finding = next(item for item in findings if item["code"] == "ADVERB_WORKS_WELL")
    assert "works well" in finding["suggestions"][0]


def test_spoken_usage_audit_catches_present_project_collocation() -> None:
    findings = offline_grammar_findings(
        "I am working on my project which I will represent to ITMO University.",
        "eng_relative_clauses",
    )

    finding = next(
        item for item in findings if item["code"] == "PRESENT_PROJECT_TO_UNIVERSITY"
    )
    assert "present to ITMO University" in finding["suggestions"][0]


def test_voice_guardrail_separates_correct_target_from_local_usage_error() -> None:
    transcript = (
        "I am working on my project which I will represent to ITMO University."
    )
    result, _ = normalize_voice_dialogue_result(
        {
            "reply_en": "Your example is correct. Next, provide another clause.",
            "feedback_ru": "Ошибок нет.",
            "grammar_score": 0.9,
            "vocabulary_score": 0.8,
            "relevance_score": 0.9,
            "correction_en": "",
            "next_goal_ru": "Продолжить.",
        },
        transcript=transcript,
        grounded_rule={"summary": "Relative clauses"},
        grammar_findings=offline_grammar_findings(transcript, "eng_relative_clauses"),
        structural_evidence=relative_clause_evidence(transcript, "eng_relative_clauses"),
    )

    assert "defining clause" in result["reply_en"]
    assert "is correct" in result["reply_en"]
    assert "present to ITMO University" in result["reply_en"]
    assert result["grammar_score"] == pytest.approx(0.72)
    assert "Целевая конструкция верна" in result["feedback_ru"]


def test_voice_guardrail_stops_repeated_exercise_after_student_complaint() -> None:
    result, needs_fallback = normalize_voice_dialogue_result(
        {
            "reply_en": "Provide a non-defining clause for the sentence again.",
            "feedback_ru": "",
            "grammar_score": 0.8,
            "vocabulary_score": 0.8,
            "relevance_score": 0.8,
            "correction_en": "",
            "next_goal_ru": "",
        },
        transcript=(
            "But I gave you the example for this sentence. Why do you ask me the second time?"
        ),
        grounded_rule={"summary": "Relative clauses"},
        history=[
            {
                "role": "assistant",
                "content": "Provide a non-defining clause for the sentence.",
            }
        ],
        service_turn=True,
    )

    assert "repeated the task" in result["reply_en"]
    assert not result["reply_en"].lower().startswith("provide")
    assert result["correction_en"] == ""
    assert needs_fallback is False


def test_voice_component_escapes_dynamic_content_and_contains_barge_in() -> None:
    rendered = voice_component_html(
        {
            "session_id": "voice-1",
            "student_id": "s01",
            "assignment_id": 1,
            "topic": "Relative clauses </h2><script>alert(1)</script>",
            "port": 8765,
        }
    )

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;/h2&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "getUserMedia" in rendered
    assert "sendJson({type:'interrupt'})" in rendered
    assert "botSpeaking" in rendered
    assert "new WebSocket(CONFIG.websocket_url)" in rendered
    assert '"websocket_url": "ws://127.0.0.1:8765"' in rendered
    assert "window.location.hostname" not in rendered
    assert 'id="manual"' in rendered
    assert "Начать реплику вручную" in rendered
    assert "manualBtn.onclick=()=>speech?endSpeech('manual'):beginSpeech(true)" in rendered
    assert "noiseFloor*1.8" in rendered
    assert "SILENCE_MS=5200" in rendered
    assert "MAX_UTTERANCE_MS=45000" in rendered
    assert "awaitingResponse=true" in rendered
    assert "if(awaitingResponse)return" in rendered
    assert "manualBtn.textContent='Реплика отправлена'" in rendered
    assert "if(manualCapture){if(now-speechStartedAt>30000)" in rendered
