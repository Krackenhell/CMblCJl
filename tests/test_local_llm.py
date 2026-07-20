from __future__ import annotations

from pathlib import Path

import pytest

from vivatrace.local_llm import LLMTrace, LocalLLM, LocalLLMError
from vivatrace.models import ProbeQuestion


def test_missing_local_model_blocks_assessment(tmp_path, monkeypatch):
    llm = LocalLLM()
    llm.model_path = tmp_path / "missing.gguf"
    llm.server_path = tmp_path / "missing-server.exe"
    monkeypatch.setattr(llm, "_is_running", lambda: False)

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
    monkeypatch.setattr(llm, "_is_running", lambda: False)

    identity = llm.identity()

    assert identity["ready"] is True
    assert identity["backend"] == "llama.cpp"
    assert identity["model_sha256"] == "deadbeef"
    assert Path(identity["model_path"]) == model
