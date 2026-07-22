from __future__ import annotations

import numpy as np
import pytest

from vivatrace.grammar import offline_grammar_findings
from vivatrace.local_llm import normalize_voice_dialogue_result
from vivatrace.voice import (
    acoustic_fluency_metrics,
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
    assert result["reply_en"].endswith("?")
    assert result["dialogue_guardrail_applied"] is True
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
    assert "manualBtn.onclick=()=>speech?endSpeech('manual'):beginSpeech()" in rendered
    assert "noiseFloor*2.4" in rendered
    assert "MAX_UTTERANCE_MS=15000" in rendered
    assert "awaitingResponse=true" in rendered
    assert "if(awaitingResponse)return" in rendered
    assert "manualBtn.textContent='Реплика отправлена'" in rendered
