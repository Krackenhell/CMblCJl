from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .database import finish_voice_session, get_assignment, init_database, save_voice_turn
from .grammar import ensure_grammar_server, offline_grammar_findings
from .local_llm import LocalLLM, LocalLLMError
from .voice import (
    VAD_MODEL,
    WHISPER_CLI,
    WHISPER_MODEL,
    acoustic_fluency_metrics,
    overall_speaking_score,
)


SAMPLE_RATE = 16_000
MAX_AUDIO_BYTES = SAMPLE_RATE * 2 * 25
INTRO = (
    "Hi! Let’s have a short B2 speaking practice. Explain one idea from today’s "
    "topic and give an English example."
)


@dataclass
class VoiceState:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    student_id: str = ""
    assignment_id: int = 0
    topic: str = "English B2"
    instructions: str = ""
    rule_id: str = ""
    configured: bool = False
    recording: bool = False
    audio: bytearray = field(default_factory=bytearray)
    history: list[dict[str, str]] = field(default_factory=list)
    generation: int = 0
    response_task: asyncio.Task[Any] | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _write_pcm_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm16)


def transcribe_pcm(pcm16: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="vivatrace-asr-") as temp_dir:
        temp = Path(temp_dir)
        wave_path = temp / "speech.wav"
        output_base = temp / "transcript"
        _write_pcm_wav(wave_path, pcm16)
        command = [
            str(WHISPER_CLI),
            "--model",
            str(WHISPER_MODEL),
            "--file",
            str(wave_path),
            "--language",
            "en",
            "--threads",
            "6",
            "--best-of",
            "2",
            "--beam-size",
            "2",
            "--no-gpu",
            "--no-fallback",
            "--suppress-nst",
            "--vad",
            "--vad-model",
            str(VAD_MODEL),
            "--vad-min-speech-duration-ms",
            "120",
            "--vad-min-silence-duration-ms",
            "250",
            "--no-prints",
            "--no-timestamps",
            "--output-txt",
            "--output-file",
            str(output_base),
        ]
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=WHISPER_CLI.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=75,
            check=False,
        )
        transcript_path = output_base.with_suffix(".txt")
        if completed.returncode != 0 or not transcript_path.exists():
            detail = (completed.stderr or completed.stdout).strip()[-500:]
            raise RuntimeError(f"Whisper не смог распознать реплику: {detail}")
        transcript = " ".join(transcript_path.read_text(encoding="utf-8").split())
    transcript = re.sub(r"\[[^]]+]|\([^)]*(?:music|silence|audio)[^)]*\)", "", transcript)
    return " ".join(transcript.split()).strip()


def synthesize_sapi(text: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="vivatrace-tts-") as temp_dir:
        temp = Path(temp_dir)
        text_path = temp / "speech.txt"
        wave_path = temp / "speech.wav"
        text_path.write_text(text, encoding="utf-8")
        escaped_text_path = str(text_path).replace("'", "''")
        escaped_wave_path = str(wave_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.SelectVoice('Microsoft Zira Desktop');$s.Rate=0;"
            f"$t=[IO.File]::ReadAllText('{escaped_text_path}',[Text.Encoding]::UTF8);"
            f"$s.SetOutputToWaveFile('{escaped_wave_path}');$s.Speak($t);$s.Dispose();"
        )
        completed = subprocess.run(  # noqa: S603
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if completed.returncode != 0 or not wave_path.exists():
            raise RuntimeError("Локальный синтез речи Windows SAPI недоступен.")
        return wave_path.read_bytes()


async def _send_json(connection: ServerConnection, state: VoiceState, payload: dict[str, Any]) -> None:
    async with state.send_lock:
        await connection.send(json.dumps(payload, ensure_ascii=False))


async def _send_audio(connection: ServerConnection, state: VoiceState, audio: bytes) -> None:
    async with state.send_lock:
        await connection.send(json.dumps({"type": "audio_start", "bytes": len(audio)}))
        await connection.send(audio)


def _trace_dict(trace: Any) -> dict[str, Any]:
    try:
        return asdict(trace)
    except TypeError:
        return {}


async def _process_utterance(
    connection: ServerConnection,
    state: VoiceState,
    pcm16: bytes,
    generation: int,
    llm: LocalLLM,
) -> None:
    try:
        await _send_json(
            connection,
            state,
            {"type": "state", "message": "Локальный Whisper распознаёт реплику…"},
        )
        transcript = await asyncio.to_thread(transcribe_pcm, pcm16)
        if generation != state.generation:
            return
        if len(re.findall(r"[A-Za-z]+", transcript)) < 2:
            await _send_json(
                connection,
                state,
                {
                    "type": "error",
                    "message": "Реплика слишком короткая или не распознана. Скажите полное предложение.",
                },
            )
            return
        metrics = acoustic_fluency_metrics(pcm16, transcript)
        grammar_findings = await asyncio.to_thread(
            offline_grammar_findings, transcript, state.rule_id
        )
        await _send_json(connection, state, {"type": "transcript", "text": transcript})
        await _send_json(
            connection,
            state,
            {"type": "state", "message": "Локальная Qwen готовит ответ и обратную связь…"},
        )
        result, trace = await asyncio.to_thread(
            llm.voice_dialogue_turn,
            rule_id=state.rule_id,
            topic=state.topic,
            instructions=state.instructions,
            history=list(state.history),
            transcript=transcript,
            acoustic_metrics=metrics,
            grammar_findings=grammar_findings,
        )
        if generation != state.generation:
            return
        reply = str(result["reply_en"])
        overall = overall_speaking_score(result, metrics)
        assessment = {
            **result,
            "metrics": metrics,
            "overall_score": overall,
            "evaluator": "local_qwen",
            "pronunciation_scored": False,
            "grammar_checker": "LanguageTool + VivaTrace rules",
            "grammar_findings": grammar_findings,
        }
        state.history.extend(
            [
                {"role": "student", "content": transcript},
                {"role": "assistant", "content": reply},
            ]
        )
        await asyncio.to_thread(
            save_voice_turn,
            session_id=state.session_id,
            student_id=state.student_id,
            assignment_id=state.assignment_id,
            student_text=transcript,
            assistant_text=reply,
            metrics=metrics,
            assessment=assessment,
            trace=_trace_dict(trace),
            overall_score=overall,
        )
        await _send_json(
            connection,
            state,
            {"type": "reply", "text": reply, "assessment": assessment},
        )
        await _send_json(
            connection,
            state,
            {"type": "state", "message": "Озвучиваю ответ. Можно перебить бота."},
        )
        audio = await asyncio.to_thread(synthesize_sapi, reply)
        if generation == state.generation:
            await _send_audio(connection, state, audio)
    except asyncio.CancelledError:
        return
    except (LocalLLMError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        if generation == state.generation:
            await _send_json(connection, state, {"type": "error", "message": str(error)})
            await _send_json(
                connection,
                state,
                {"type": "state", "message": "Слушаю следующую реплику."},
            )


async def _cancel_response(state: VoiceState) -> None:
    state.generation += 1
    if state.response_task and not state.response_task.done():
        state.response_task.cancel()
    state.response_task = None


async def _configure(
    connection: ServerConnection,
    state: VoiceState,
    payload: dict[str, Any],
) -> None:
    student_id = str(payload.get("student_id") or "")
    assignment_id = int(payload.get("assignment_id") or 0)
    assignment = get_assignment(assignment_id)
    if not student_id or not assignment:
        raise ValueError("Не удалось связать голосовую сессию со студентом и заданием.")
    state.session_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(payload.get("session_id") or ""))[
        :80
    ] or str(uuid4())
    state.student_id = student_id
    state.assignment_id = assignment_id
    state.topic = str(payload.get("topic") or assignment.get("topic") or "English B2")[:180]
    state.instructions = str(assignment.get("instructions") or "")[:900]
    state.rule_id = str((assignment.get("skill_ids") or [""])[0])
    state.configured = True
    await _send_json(
        connection,
        state,
        {"type": "state", "message": "Слушаю. Начните говорить по-английски."},
    )
    generation = state.generation
    audio = await asyncio.to_thread(synthesize_sapi, INTRO)
    if generation == state.generation:
        await _send_audio(connection, state, audio)


async def handle_connection(connection: ServerConnection) -> None:
    state = VoiceState()
    llm = LocalLLM()
    try:
        async for message in connection:
            if isinstance(message, bytes):
                if state.recording and len(state.audio) < MAX_AUDIO_BYTES:
                    state.audio.extend(message[: MAX_AUDIO_BYTES - len(state.audio)])
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            message_type = str(payload.get("type") or "")
            if message_type == "configure":
                try:
                    await _configure(connection, state, payload)
                except (ValueError, KeyError) as error:
                    await _send_json(connection, state, {"type": "error", "message": str(error)})
            elif message_type in {"interrupt", "speech_start"}:
                await _cancel_response(state)
                if message_type == "speech_start":
                    state.audio.clear()
                    state.recording = True
            elif message_type == "speech_end" and state.recording and state.configured:
                state.recording = False
                pcm16 = bytes(state.audio)
                state.audio.clear()
                if len(pcm16) < SAMPLE_RATE * 2 // 3:
                    await _send_json(
                        connection,
                        state,
                        {"type": "error", "message": "Реплика короче трети секунды, попробуйте ещё раз."},
                    )
                    continue
                generation = state.generation
                state.response_task = asyncio.create_task(
                    _process_utterance(connection, state, pcm16, generation, llm)
                )
            elif message_type == "finish":
                await _cancel_response(state)
                if state.configured:
                    await asyncio.to_thread(finish_voice_session, state.session_id)
    except ConnectionClosed:
        pass
    finally:
        await _cancel_response(state)
        if state.configured:
            await asyncio.to_thread(finish_voice_session, state.session_id)


async def run_server(port: int) -> None:
    init_database()
    await asyncio.to_thread(ensure_grammar_server)
    async with serve(
        handle_connection,
        "127.0.0.1",
        port,
        max_size=2_000_000,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
    ):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(run_server(args.port))


if __name__ == "__main__":
    main()
