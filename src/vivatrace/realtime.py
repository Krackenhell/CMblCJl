from __future__ import annotations

import hashlib
import json
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .database import finish_voice_session, get_assignment, list_students, save_voice_turn
from .rulebook import load_rulebook, rules_for_assignment
from .voice import is_assessable_spoken_turn, overall_speaking_score


REALTIME_PORT = 8766
REALTIME_MODEL = "gpt-realtime-2.1"
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
ASSESSMENT_MODEL = "gpt-5.6-luna"
OPENAI_API_ROOT = "https://api.openai.com/v1"

_SERVER_LOCK = threading.Lock()
_SERVER: _RealtimeHTTPServer | None = None
_SERVER_THREAD: threading.Thread | None = None


def _bridge_token_for_key(api_key: str) -> str:
    """Build a stable, non-reversible token for the local browser bridge."""
    return hashlib.sha256(f"vivatrace-realtime-bridge-v1:{api_key}".encode()).hexdigest()


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return default


def _safe_error_message(payload: bytes, status: int) -> str:
    message = "OpenAI API отклонил запрос."
    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
        candidate = data.get("error", {}).get("message")
        if isinstance(candidate, str) and candidate.strip():
            message = candidate.strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    if status == 401:
        return "API-ключ не принят. Проверьте ключ и доступ проекта к Realtime API."
    if status == 429:
        return "Достигнут лимит OpenAI API или на проекте нет доступной квоты."
    return f"{message} (HTTP {status})"


def _request_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout: float = 60,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read()
            response_headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(_safe_error_message(body, exc.code)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к OpenAI API: {exc.reason}") from exc
    try:
        return json.loads(body.decode("utf-8")), response_headers
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenAI API вернул ответ в неожиданном формате.") from exc


def _response_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    fragments: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
    return "\n".join(fragments)


def _grounded_rule(assignment: dict[str, Any]) -> dict[str, Any]:
    rules = rules_for_assignment(assignment, load_rulebook())
    return rules[0] if rules else {
        "title": assignment.get("topic") or "English B2",
        "summary": "Use accurate B2 English that fits the meaning and context.",
        "principles": [],
        "examples": [],
    }


def build_realtime_session_config(assignment: dict[str, Any]) -> dict[str, Any]:
    rule = _grounded_rule(assignment)
    principles = " | ".join(str(item) for item in rule.get("principles", [])[:4])
    examples = " | ".join(str(item) for item in rule.get("examples", [])[:3])
    instructions = f"""
You are VivaTrace, a warm but rigorous B2 English speaking tutor.

LESSON TOPIC: {assignment.get('topic') or 'English B2'}
TASK CONTEXT: {assignment.get('instructions') or assignment.get('title') or ''}
GROUNDED RULE: {rule.get('summary') or ''}
KEY PRINCIPLES: {principles}
VALID EXAMPLES: {examples}

Run a natural speech-to-speech conversation in English.
- Listen to the audio itself; the auxiliary transcript is only for captions.
- Before replying, classify the turn as: lesson answer, unfinished thought, or a service
  request to you. Answer service requests (for example "Are you listening?") directly and
  invite the student to continue; never treat them as lesson answers.
- React to the student's exact meaning and wording. Never give generic praise such as
  "useful example" when the utterance is vague, off-topic, or unclear.
- Do not invent restrictions, errors, or facts absent from the task and the student's words.
- If you hear a clear grammar or vocabulary error, briefly recast the student's own sentence
  and ask them to say the corrected version once. Do not turn every error into a theory question.
- If the answer is correct, name the exact successful choice and request one concrete next
  example only when more evidence is actually needed.
- Treat accent, hesitation, punctuation, capitalization, and likely ASR spelling noise as
  separate from grammar. Ask for clarification when the audio is genuinely ambiguous.
- Never ask the stock question "Why did you choose that form, and how would another form change
  the meaning?" Ask about an alternative only when the student's exact sentence creates a real
  meaning contrast.
- Keep each spoken turn to at most 2 short sentences and about 30 spoken words. A question is
  optional and there must never be more than one. Use a natural B2 pace.
- The student may interrupt you. Stop immediately, listen, and continue from their new point.
- On the first turn, greet the student, name the lesson topic, and ask for one original example.
""".strip()
    transcription_prompt = (
        f"English B2 lesson about {assignment.get('topic')}. "
        f"Likely terminology: {rule.get('title')}; {principles}; {examples}. "
        "Preserve the speaker's actual words; do not silently correct grammar."
    )[:900]
    return {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "output_modalities": ["audio"],
        "instructions": instructions,
        "max_output_tokens": 120,
        "audio": {
            "input": {
                "noise_reduction": {"type": "far_field"},
                "transcription": {
                    "model": TRANSCRIPTION_MODEL,
                    "language": "en",
                    "prompt": transcription_prompt,
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 400,
                    "silence_duration_ms": 5200,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": "marin", "speed": 1.0},
        },
        "tracing": {
            "workflow_name": "VivaTrace Realtime speaking",
            "group_id": str(assignment.get("topic_key") or "english-b2"),
            "metadata": {"assignment_id": str(assignment.get("id") or "")},
        },
    }


def _multipart_session_body(sdp: str, session: dict[str, Any]) -> tuple[bytes, str]:
    boundary = f"vivatrace-{secrets.token_hex(16)}"
    parts = [
        (
            "sdp",
            "application/sdp",
            sdp.encode("utf-8"),
        ),
        (
            "session",
            "application/json",
            json.dumps(session, ensure_ascii=False).encode("utf-8"),
        ),
    ]
    body = bytearray()
    for name, content_type, content in parts:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n'.encode())
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


def create_realtime_call(
    *,
    api_key: str,
    assignment: dict[str, Any],
    student_id: str,
    sdp: str,
) -> tuple[str, str]:
    body, boundary = _multipart_session_body(sdp, build_realtime_session_config(assignment))
    safety_id = hashlib.sha256(f"vivatrace:{student_id}".encode()).hexdigest()
    request = urllib.request.Request(
        f"{OPENAI_API_ROOT}/realtime/calls",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "OpenAI-Safety-Identifier": safety_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
            answer = response.read().decode("utf-8")
            location = response.headers.get("Location", "")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(_safe_error_message(body, exc.code)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к OpenAI Realtime: {exc.reason}") from exc
    if not answer.lstrip().startswith("v="):
        raise RuntimeError("Realtime API не вернул корректный SDP-ответ.")
    return answer, location.rsplit("/", 1)[-1] if location else ""


ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grammar_score": {"type": "number", "minimum": 0, "maximum": 1},
        "vocabulary_score": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
        "feedback_ru": {"type": "string"},
        "correction_en": {"type": "string"},
        "next_goal_ru": {"type": "string"},
        "evidence_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "grammar_score",
        "vocabulary_score",
        "relevance_score",
        "feedback_ru",
        "correction_en",
        "next_goal_ru",
        "evidence_quote",
        "confidence",
    ],
    "additionalProperties": False,
}


def assess_realtime_turn(
    *,
    api_key: str,
    assignment: dict[str, Any],
    student_text: str,
    assistant_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rule = _grounded_rule(assignment)
    payload = {
        "model": ASSESSMENT_MODEL,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 700,
        "input": [
            {
                "role": "system",
                "content": (
                    "Ты независимый оценщик устной речи уровня B2. Оцени только буквальный "
                    "транскрипт студента по заданной теме и проверенной базе правила. Не штрафуй "
                    "за пунктуацию, регистр, акцент, паузы и вероятный шум распознавания. Не выдумывай "
                    "ошибки и ограничения. Любое замечание привяжи к точной короткой цитате. Если "
                    "ошибки нет, correction_en оставь пустым. Ответ и обратная связь преподавателя "
                    "не являются доказательством ошибки студента. Пиши обратную связь по-русски."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "topic": assignment.get("topic"),
                        "task_context": assignment.get("instructions"),
                        "grounded_rule": rule,
                        "student_transcript": student_text,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "speaking_assessment",
                "schema": ASSESSMENT_SCHEMA,
                "strict": True,
            }
        },
    }
    response, headers = _request_json(
        f"{OPENAI_API_ROOT}/responses",
        api_key=api_key,
        payload=payload,
        timeout=60,
    )
    output = _response_output_text(response)
    try:
        assessment = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Модель оценки вернула невалидный структурированный ответ.") from exc
    for key in ("grammar_score", "vocabulary_score", "relevance_score", "confidence"):
        assessment[key] = _clamp(assessment.get(key))
    assessment["scoring_available"] = True
    assessment["engine"] = "openai_responses"
    trace = {
        "trace_id": str(response.get("id") or headers.get("x-request-id") or ""),
        "engine": "openai",
        "dialogue_model": REALTIME_MODEL,
        "transcription_model": TRANSCRIPTION_MODEL,
        "assessment_model": ASSESSMENT_MODEL,
        "grounding_rule_id": rule.get("id"),
    }
    return assessment, trace


def _fluency_proxy(student_text: str, duration_seconds: Any) -> dict[str, Any]:
    words = [word for word in student_text.split() if word.strip(".,!?;:\"'")]
    duration = max(0.1, min(180.0, float(duration_seconds or 0.1)))
    words_per_minute = len(words) / duration * 60
    if 85 <= words_per_minute <= 180:
        fluency = 1.0
    elif 60 <= words_per_minute <= 210:
        fluency = 0.72
    elif len(words) >= 4:
        fluency = 0.45
    else:
        fluency = 0.2
    return {
        "duration_seconds": round(duration, 2),
        "word_count": len(words),
        "words_per_minute": round(words_per_minute, 1),
        "fluency_score": round(fluency, 3),
        "metric_type": "webrtc_turn_timing_proxy",
    }


def persist_realtime_turn(
    *,
    api_key: str,
    session_id: str,
    student_id: str,
    assignment_id: int,
    student_text: str,
    assistant_text: str,
    duration_seconds: Any,
    realtime_call_id: str = "",
    client_turn_id: str = "",
) -> dict[str, Any]:
    assignment = get_assignment(assignment_id)
    if not is_assessable_spoken_turn(student_text):
        return {
            "skipped": True,
            "reason": "service_turn",
            "assessment": {"scoring_available": False},
        }
    metrics = _fluency_proxy(student_text, duration_seconds)
    try:
        assessment, trace = assess_realtime_turn(
            api_key=api_key,
            assignment=assignment,
            student_text=student_text,
            assistant_text=assistant_text,
        )
    except RuntimeError as exc:
        assessment = {
            "grammar_score": 0.0,
            "vocabulary_score": 0.0,
            "relevance_score": 0.0,
            "feedback_ru": "Диалог сохранён; отдельная оценка временно недоступна.",
            "correction_en": "",
            "next_goal_ru": "Продолжить устный диалог.",
            "evidence_quote": student_text[:160],
            "confidence": 0.0,
            "scoring_available": False,
            "scoring_error": str(exc),
            "engine": "openai_realtime_unscored",
        }
        trace = {
            "engine": "openai",
            "dialogue_model": REALTIME_MODEL,
            "transcription_model": TRANSCRIPTION_MODEL,
            "assessment_model": ASSESSMENT_MODEL,
            "assessment_status": "unavailable",
        }
    trace["realtime_call_id"] = realtime_call_id
    trace["client_turn_id"] = client_turn_id
    overall = (
        overall_speaking_score(assessment, metrics)
        if assessment.get("scoring_available")
        else 0.0
    )
    save_voice_turn(
        session_id=session_id,
        student_id=student_id,
        assignment_id=assignment_id,
        student_text=student_text,
        assistant_text=assistant_text,
        metrics=metrics,
        assessment=assessment,
        trace=trace,
        overall_score=overall,
    )
    return {"assessment": assessment, "metrics": metrics, "overall_score": overall, "trace": trace}


class _RealtimeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], api_key: str):
        super().__init__(address, _RealtimeHandler)
        self.api_key = api_key
        self.bridge_token = _bridge_token_for_key(api_key)
        self.key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:12]

    def update_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        self.bridge_token = _bridge_token_for_key(api_key)
        self.key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:12]


class _RealtimeHandler(BaseHTTPRequestHandler):
    server: _RealtimeHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _set_cors(self) -> None:
        origin = self.headers.get("Origin", "")
        allowed = origin if origin.startswith(("http://127.0.0.1:", "http://localhost:")) else "null"
        self.send_header("Access-Control-Allow-Origin", allowed)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Expose-Headers", "X-Realtime-Call-Id")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized_query(self) -> dict[str, list[str]] | None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if query.get("bridge_token", [""])[0] != self.server.bridge_token:
            self._send_json(403, {"error": "Локальный Realtime-шлюз отклонил запрос."})
            return None
        return query

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/health":
            self._send_json(404, {"error": "Not found"})
            return
        self._send_json(
            200,
            {"ready": bool(self.server.api_key), "model": REALTIME_MODEL, "version": 1},
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        query = self._authorized_query()
        if query is None:
            return
        try:
            content_length = min(int(self.headers.get("Content-Length", "0")), 2_000_000)
            body = self.rfile.read(content_length)
            if path == "/session":
                self._handle_session(query, body)
            elif path == "/turn":
                self._handle_turn(body)
            elif path == "/finish":
                self._handle_finish(body)
            else:
                self._send_json(404, {"error": "Not found"})
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(502, {"error": str(exc)})
        except Exception:
            self._send_json(500, {"error": "Внутренняя ошибка Realtime-шлюза."})

    def _handle_session(self, query: dict[str, list[str]], body: bytes) -> None:
        student_id = query.get("student_id", [""])[0]
        assignment_id = int(query.get("assignment_id", ["0"])[0])
        if student_id not in {item["id"] for item in list_students()}:
            raise ValueError("Неизвестный профиль студента.")
        assignment = get_assignment(assignment_id)
        sdp = body.decode("utf-8")
        if not sdp.lstrip().startswith("v="):
            raise ValueError("Браузер не передал корректное WebRTC-предложение.")
        answer, call_id = create_realtime_call(
            api_key=self.server.api_key,
            assignment=assignment,
            student_id=student_id,
            sdp=sdp,
        )
        encoded = answer.encode("utf-8")
        self.send_response(201)
        self._set_cors()
        self.send_header("Content-Type", "application/sdp")
        self.send_header("Content-Length", str(len(encoded)))
        if call_id:
            self.send_header("X-Realtime-Call-Id", call_id)
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_turn(self, body: bytes) -> None:
        data = json.loads(body.decode("utf-8"))
        student_text = str(data.get("student_text") or "").strip()
        assistant_text = str(data.get("assistant_text") or "").strip()
        if len(student_text) < 2 or len(assistant_text) < 2:
            raise ValueError("Недостаточно текста для сохранения реплики.")
        result = persist_realtime_turn(
            api_key=self.server.api_key,
            session_id=str(data["session_id"]),
            student_id=str(data["student_id"]),
            assignment_id=int(data["assignment_id"]),
            student_text=student_text[:4000],
            assistant_text=assistant_text[:4000],
            duration_seconds=data.get("duration_seconds"),
            realtime_call_id=str(data.get("realtime_call_id") or ""),
            client_turn_id=str(data.get("client_turn_id") or "")[:80],
        )
        self._send_json(200, result)

    def _handle_finish(self, body: bytes) -> None:
        data = json.loads(body.decode("utf-8"))
        finish_voice_session(str(data["session_id"]))
        self._send_json(200, {"ok": True})


def ensure_realtime_server(api_key: str, port: int = REALTIME_PORT) -> dict[str, Any]:
    global _SERVER, _SERVER_THREAD
    clean_key = api_key.strip()
    if not clean_key:
        return {"ready": False, "error": "API-ключ не указан."}
    with _SERVER_LOCK:
        if _SERVER is not None and _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            _SERVER.update_api_key(clean_key)
            return {
                "ready": True,
                "port": _SERVER.server_port,
                "bridge_token": _SERVER.bridge_token,
                "model": REALTIME_MODEL,
                "transcription_model": TRANSCRIPTION_MODEL,
                "assessment_model": ASSESSMENT_MODEL,
            }
        try:
            _SERVER = _RealtimeHTTPServer(("127.0.0.1", port), clean_key)
        except OSError as exc:
            return {
                "ready": False,
                "error": f"Не удалось запустить Realtime-шлюз на порту {port}: {exc}",
            }
        _SERVER_THREAD = threading.Thread(
            target=_SERVER.serve_forever,
            name="vivatrace-openai-realtime",
            daemon=True,
        )
        _SERVER_THREAD.start()
        return {
            "ready": True,
            "port": _SERVER.server_port,
            "bridge_token": _SERVER.bridge_token,
            "model": REALTIME_MODEL,
            "transcription_model": TRANSCRIPTION_MODEL,
            "assessment_model": ASSESSMENT_MODEL,
        }
