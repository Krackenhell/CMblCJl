from __future__ import annotations

import json

import pytest

from meaning_trainer.realtime import (
    ASSESSMENT_MODEL,
    REALTIME_MODEL,
    TRANSCRIPTION_MODEL,
    _bridge_token_for_key,
    _multipart_session_body,
    _response_output_text,
    _safe_error_message,
    assess_realtime_turn,
    build_realtime_session_config,
)
from meaning_trainer.voice import realtime_voice_component_html


@pytest.fixture
def relative_assignment() -> dict:
    return {
        "id": 17,
        "title": "Relative clauses in context",
        "topic": "Определительные придаточные",
        "topic_key": "eng_relative_clauses",
        "instructions": "Combine the ideas and explain the meaning.",
        "skill_ids": ["eng_relative_clauses"],
    }


def test_realtime_session_is_grounded_and_full_duplex(relative_assignment: dict) -> None:
    session = build_realtime_session_config(relative_assignment)

    assert session["model"] == REALTIME_MODEL
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["transcription"]["model"] == TRANSCRIPTION_MODEL
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 400,
        "silence_duration_ms": 5200,
        "create_response": True,
        "interrupt_response": True,
    }
    assert "Defining clause" in session["instructions"]
    assert "Never give generic praise" in session["instructions"]
    assert session["tracing"]["workflow_name"] == "Meaning Realtime speaking"


def test_realtime_component_uses_webrtc_without_exposing_api_key() -> None:
    rendered = realtime_voice_component_html(
        {
            "session_id": "voice-cloud-1",
            "student_id": "s01",
            "assignment_id": 17,
            "topic": "Relative clauses </h2><script>alert(1)</script>",
            "backend_url": "http://127.0.0.1:8766",
            "bridge_token": "temporary-local-bridge-token",
        }
    )

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;/h2&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "new RTCPeerConnection()" in rendered
    assert "pc.addTrack" in rendered
    assert "createDataChannel('oai-events')" in rendered
    assert "input_audio_buffer.speech_started" in rendered
    assert "response.output_audio_transcript.done" in rendered
    assert "conversation.item.input_audio_transcription.completed" in rendered
    assert "client_turn_id:turn.id" in rendered
    assert "saveChain=saveChain.then(()=>saveTurn(turn))" in rendered
    assert "isServiceTurn(text)" in rendered
    assert "durationByItemId" in rendered
    assert "studentQueue" not in rendered
    assert "assistantQueue" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "Authorization" not in rendered
    assert "temporary-local-bridge-token" in rendered


def test_bridge_token_is_stable_across_gateway_restart() -> None:
    first = _bridge_token_for_key("sk-test-one")
    restarted = _bridge_token_for_key("sk-test-one")
    different_key = _bridge_token_for_key("sk-test-two")

    assert first == restarted
    assert first != different_key
    assert "sk-test-one" not in first


def test_multipart_contains_sdp_and_session(relative_assignment: dict) -> None:
    session = build_realtime_session_config(relative_assignment)
    body, boundary = _multipart_session_body("v=0\r\ns=meaning_trainer", session)
    decoded = body.decode("utf-8")

    assert f"--{boundary}" in decoded
    assert 'name="sdp"' in decoded
    assert "Content-Type: application/sdp" in decoded
    assert "v=0\r\ns=meaning_trainer" in decoded
    assert 'name="session"' in decoded
    assert f'"model": "{REALTIME_MODEL}"' in decoded


def test_response_output_text_reads_raw_responses_payload() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"grammar_score": 0.9}'}],
            }
        ]
    }
    assert _response_output_text(payload) == '{"grammar_score": 0.9}'


def test_assessment_uses_structured_outputs(monkeypatch, relative_assignment: dict) -> None:
    captured: dict = {}
    answer = {
        "grammar_score": 0.9,
        "vocabulary_score": 0.8,
        "relevance_score": 0.95,
        "feedback_ru": "Форма who правильно относится к человеку.",
        "correction_en": "",
        "next_goal_ru": "Добавить non-defining пример.",
        "evidence_quote": "John, who is a player",
        "confidence": 0.88,
    }

    def fake_request(url, *, api_key, payload, timeout, headers=None):
        captured.update({"url": url, "api_key": api_key, "payload": payload})
        return {
            "id": "resp_test",
            "output": [
                {"content": [{"type": "output_text", "text": json.dumps(answer)}]}
            ],
        }, {}

    monkeypatch.setattr("meaning_trainer.realtime._request_json", fake_request)
    assessment, trace = assess_realtime_turn(
        api_key="sk-test-not-real",
        assignment=relative_assignment,
        student_text="I met John, who is a professional player.",
        assistant_text="Good. Why did you use commas?",
    )

    assert captured["payload"]["model"] == ASSESSMENT_MODEL
    assert captured["payload"]["text"]["format"]["type"] == "json_schema"
    assert captured["payload"]["text"]["format"]["strict"] is True
    assessment_input = captured["payload"]["input"][1]["content"]
    assert '"student_transcript"' in assessment_input
    assert '"tutor_reply"' not in assessment_input
    assert assessment["grammar_score"] == pytest.approx(0.9)
    assert assessment["scoring_available"] is True
    assert trace["trace_id"] == "resp_test"
    assert trace["dialogue_model"] == REALTIME_MODEL


def test_api_errors_are_safe_and_actionable() -> None:
    assert "API-ключ" in _safe_error_message(b"{}", 401)
    assert "квот" in _safe_error_message(b"{}", 429)
    assert "sk-secret" not in _safe_error_message(
        b'{"error":{"message":"request failed"}}', 500
    )
