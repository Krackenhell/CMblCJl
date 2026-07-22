from __future__ import annotations

import argparse
import io
import json
import time
import wave

import numpy as np
from websockets.sync.client import connect

from meaning_trainer.voice_server import synthesize_sapi


def sapi_wave_to_pcm16(wave_bytes: bytes, target_rate: int = 16_000) -> bytes:
    with wave.open(io.BytesIO(wave_bytes), "rb") as source:
        source_rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(
            source.readframes(source.getnframes()), dtype="<i2"
        ).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    target_length = round(len(samples) * target_rate / source_rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, target_length),
        np.arange(len(samples)),
        samples,
    )
    return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()


def run_smoke(
    sentence: str,
    *,
    student_id: str,
    assignment_id: int,
    session_id: str,
    port: int,
) -> dict:
    pcm16 = sapi_wave_to_pcm16(synthesize_sapi(sentence))
    started = time.monotonic()
    transcript = ""
    reply: dict = {}
    with connect(
        f"ws://127.0.0.1:{port}",
        open_timeout=5,
        close_timeout=2,
        max_size=2_000_000,
    ) as websocket:
        websocket.send(
            json.dumps(
                {
                    "type": "configure",
                    "session_id": session_id,
                    "student_id": student_id,
                    "assignment_id": assignment_id,
                }
            )
        )
        while not isinstance(websocket.recv(timeout=45), bytes):
            pass
        websocket.send(json.dumps({"type": "speech_start"}))
        for offset in range(0, len(pcm16), 8192):
            websocket.send(pcm16[offset : offset + 8192])
        websocket.send(json.dumps({"type": "speech_end"}))
        while not reply:
            message = websocket.recv(timeout=180)
            if not isinstance(message, str):
                continue
            payload = json.loads(message)
            if payload.get("type") == "error":
                raise RuntimeError(str(payload.get("message")))
            if payload.get("type") == "transcript":
                transcript = str(payload.get("text") or "")
            if payload.get("type") == "reply":
                reply = payload
        websocket.send(json.dumps({"type": "finish"}))
    return {
        "input": sentence,
        "transcript": transcript,
        "reply": reply.get("text"),
        "assessment": reply.get("assessment"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Сквозная проверка локального голосового диалога")
    parser.add_argument("sentence")
    parser.add_argument("--student", default="s01")
    parser.add_argument("--assignment", type=int, default=92)
    parser.add_argument("--session", default="voice-smoke")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    result = run_smoke(
        args.sentence,
        student_id=args.student,
        assignment_id=args.assignment,
        session_id=args.session,
        port=args.port,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
