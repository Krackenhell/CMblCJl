from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Evidence, ProbeQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
DEFAULT_SERVER_PATH = PROJECT_ROOT / "tools" / "llama" / "llama-server.exe"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "local-model-manifest.json"


class LocalLLMError(RuntimeError):
    """Raised when the required local model cannot produce a verified result."""


@dataclass(frozen=True)
class LLMTrace:
    trace_id: str
    backend: str
    model: str
    model_sha256: str
    stage: str
    duration_ms: int
    created_at: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LocalLLM:
    """Strict local-LLM gateway backed by llama.cpp.

    There is deliberately no heuristic or remote-API fallback. If the model is
    absent or returns invalid JSON, the assessment stops instead of fabricating
    a grade.
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080").rstrip("/")
        self.model_path = Path(os.getenv("LOCAL_LLM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self.server_path = Path(os.getenv("LOCAL_LLM_SERVER_PATH", str(DEFAULT_SERVER_PATH)))
        self.manifest_path = Path(
            os.getenv("LOCAL_LLM_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH))
        )
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-7B-Instruct-Q4_K_M")

    def identity(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            # Windows PowerShell 5 writes UTF-8 JSON with a BOM by default.
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        running = self._is_running()
        return {
            "ready": running or (self.model_path.exists() and self.server_path.exists()),
            "running": running,
            "backend": "llama.cpp",
            "model": manifest.get("model", self.model_name),
            "model_sha256": manifest.get("model_sha256", "не вычислен"),
            "model_path": str(self.model_path),
            "server_path": str(self.server_path),
            "setup_command": r"powershell -ExecutionPolicy Bypass -File scripts\setup_local_llm.ps1",
        }

    def ensure_available(self) -> None:
        if self._is_running():
            return
        if not self.model_path.exists() or not self.server_path.exists():
            raise LocalLLMError(
                "Локальная LLM не установлена. Запустите scripts\\setup_local_llm.ps1."
            )
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout = (log_dir / "local-llm.stdout.log").open("a", encoding="utf-8")
        stderr = (log_dir / "local-llm.stderr.log").open("a", encoding="utf-8")
        command = [
            str(self.server_path),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            self.base_url.rsplit(":", 1)[-1],
            "--ctx-size",
            "6144",
            "--threads",
            str(max((os.cpu_count() or 4) - 1, 2)),
            "--jinja",
            "--no-webui",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(  # noqa: S603
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self._is_running():
                return
            time.sleep(1)
        raise LocalLLMError("Локальная модель не запустилась за 120 секунд. Проверьте logs/.")

    def assess_submission(
        self,
        assignment: dict[str, Any],
        answer: str,
        skill_names: dict[str, str],
    ) -> tuple[dict[str, Any], list[LLMTrace]]:
        grade_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "submission_score",
                "is_correct",
                "feedback",
                "mode",
                "skill_results",
            ],
            "properties": {
                "submission_score": {"type": "number", "minimum": 0, "maximum": 1},
                "is_correct": {"type": "boolean"},
                "feedback": {"type": "string"},
                "mode": {"type": "string", "enum": ["viva", "diagnostic"]},
                "skill_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["skill_id", "score", "diagnosis"],
                        "properties": {
                            "skill_id": {"type": "string", "enum": assignment["skill_ids"]},
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                            "diagnosis": {"type": "string"},
                        },
                    },
                },
            },
        }
        rubric = assignment.get("rubric") or {}
        system = (
            "Ты строгий методист университета. Оценивай фактическую правильность, а не длину "
            "и уверенность текста. Бессмысленный, нерелевантный или тавтологический ответ получает "
            "0–0.10. is_correct=true только при score >= 0.75 и отсутствии критической ошибки. "
            "Сначала по пунктам сопоставь ответ студента с reference_answer и criteria из входных данных, "
            "затем выставляй общий балл. Reference_answer — основание проверки: не объявляй правильную "
            "форму ошибочной. Смысловые эквиваленты, полные и сокращённые формы (например, has not и "
            "hasn't) считаются одинаковыми. Если все запрошенные пункты совпадают с эталоном и объяснены, "
            "submission_score должен быть не ниже 0.85, а is_correct=true. submission_score, is_correct "
            "и skill_results не должны противоречить друг другу. Для каждого skill_id верни ровно один "
            "skill_result. Все feedback, diagnosis и вопросы должны быть законченными предложениями. "
            "Если все пункты правильны, прямо напиши, что содержательных ошибок нет; не придумывай "
            "оговорки, неточности или ошибки, которых нет в ответе. "
            "Если решение правильное, mode=viva. Если решение неправильное, mode=diagnostic. "
            "Пиши объяснения по-русски, а примеры "
            "английского оставляй на английском. Верни только JSON по заданной схеме."
        )
        user_payload = {
            "subject": assignment.get("subject"),
            "topic": assignment["topic"],
            "task": assignment["instructions"],
            "rubric": rubric,
            "skills": {skill_id: skill_names[skill_id] for skill_id in assignment["skill_ids"]},
            "student_answer": answer,
        }
        result, grade_trace = self._call_json(
            "проверка задания", system, user_payload, grade_schema, 700
        )
        question_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["skill_id", "text", "purpose"],
                        "properties": {
                            "skill_id": {"type": "string", "enum": assignment["skill_ids"]},
                            "text": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                    },
                }
            },
        }
        question_system = (
            "Ты проводишь короткую устную проверку после учебного задания. Используй только переданные "
            "задание, рубрику и уже выполненную LLM-оценку. Верни ровно три законченных вопроса. "
            "Если mode=viva, проверь причины выбора, перенос правила и самостоятельное понимание, не проси "
            "просто повторить ответ. Запрещено копировать исходное предложение с пропуском или снова просить "
            "вставить ту же форму. Каждый viva-вопрос должен требовать объяснения: например, «Почему здесь "
            "нельзя использовать альтернативу?», «Что изменится, если поменять временной маркер?», «Приведи "
            "новый пример того же правила». Если mode=diagnostic, локализуй конкретный пробел и двигайся от простого "
            "правила к применению с подсказкой. Вопросы и purpose пиши по-русски, английские примеры оставляй "
            "на английском. Верни только JSON по схеме."
        )
        question_payload = {**user_payload, "llm_assessment": result}
        question_result, question_trace = self._call_json(
            "формирование viva или диагностики",
            question_system,
            question_payload,
            question_schema,
            650,
        )
        result["questions"] = [
            ProbeQuestion(
                id=f"local-{question_trace.trace_id}-{index}",
                skill_id=item["skill_id"],
                text=item["text"],
                purpose=item["purpose"],
                expected_concepts=(),
            )
            for index, item in enumerate(question_result["questions"], start=1)
        ]
        return result, [grade_trace, question_trace]

    def evaluate_answer(
        self,
        assignment: dict[str, Any],
        question: ProbeQuestion,
        answer: str,
    ) -> tuple[Evidence, LLMTrace]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "confidence", "rationale", "misconception"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "misconception": {"type": ["string", "null"]},
            },
        }
        system = (
            "Ты строгий оценщик ответа на проверочный вопрос. Оцени только релевантное предметное "
            "понимание. Совпадение отдельных слов, перефразирование вопроса, фразы «не знаю», "
            "«вся информация», «нужно воспроизвести» и случайный текст не являются доказательством "
            "знания и должны получать score 0–0.10. Частично верная причинная связь — 0.3–0.6; "
            "полный корректный ответ — 0.75–1.0. rationale должно явно назвать, что верно, чего "
            "не хватает и почему выставлен балл. rationale и misconception пиши строго по-русски; "
            "не используй китайский или другой язык, кроме английских примеров из задания. "
            "Верни только JSON."
        )
        payload = {
            "task": assignment["instructions"],
            "rubric": assignment.get("rubric") or {},
            "question": asdict(question),
            "student_answer": answer,
        }
        result, trace = self._call_json("оценка ответа viva", system, payload, schema, 500)
        evidence = Evidence(
            skill_id=question.skill_id,
            score=round(float(result["score"]), 3),
            confidence=round(float(result["confidence"]), 3),
            quote=answer.strip()[:360],
            rationale=str(result["rationale"]),
            misconception=result.get("misconception"),
            source="local_llm",
            evaluator_model=trace.model,
            trace_id=trace.trace_id,
        )
        return evidence, trace

    def finalize_learning(
        self,
        assignment: dict[str, Any],
        assessment: dict[str, Any],
        evidence: list[Evidence],
        cohort_context: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], LLMTrace]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["branch", "student_activity", "teacher_recommendation"],
            "properties": {
                "branch": {"type": "string", "enum": ["transfer", "remediation"]},
                "student_activity": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "title",
                        "instructions",
                        "why",
                        "explanation",
                        "worked_example",
                        "practice_task",
                        "success_criteria",
                    ],
                    "properties": {
                        "title": {"type": "string"},
                        "instructions": {"type": "string"},
                        "why": {"type": "string"},
                        "explanation": {"type": "string"},
                        "worked_example": {"type": "string"},
                        "practice_task": {"type": "string"},
                        "success_criteria": {"type": "string"},
                    },
                },
                "teacher_recommendation": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["focus_topic", "reason", "lesson_plan", "evidence_summary"],
                    "properties": {
                        "focus_topic": {"type": "string"},
                        "reason": {"type": "string"},
                        "lesson_plan": {"type": "string"},
                        "evidence_summary": {"type": "string"},
                    },
                },
            },
        }
        system = (
            "Ты адаптивный методист. На основании только переданных оценок создай следующий шаг. "
            "Если исходное задание и viva подтверждают понимание, branch=transfer: новое задание "
            "закрепляет тот же навык в другом контексте и не копирует исходное. Иначе "
            "branch=remediation: дай короткое объяснение, разобранный пример и посильную повторную "
            "практику по точному пробелу в трёх отдельных полях explanation, worked_example и "
            "practice_task. Не пиши обещание «разберём пример» — приведи сам пример и его разбор. "
            "Перед ответом перепроверь worked_example и practice_task по reference_answer, criteria и "
            "common_errors из рубрики. Пример обязан быть предметно правильным; никогда не выдавай шаблон "
            "из common_errors за правильный. "
            "Затем предложи преподавателю фокус следующего занятия "
            "по данным всей доступной группы. Не придумывай студентов или результаты. Верни JSON."
        )
        payload = {
            "assignment": {
                "subject": assignment.get("subject"),
                "topic": assignment["topic"],
                "instructions": assignment["instructions"],
                "rubric": assignment.get("rubric") or {},
            },
            "submission_assessment": {
                key: value for key, value in assessment.items() if key != "questions"
            },
            "viva_evidence": [asdict(item) for item in evidence],
            "cohort_latest_results": cohort_context,
        }
        return self._call_json("адаптивный маршрут и план пары", system, payload, schema, 1100)

    def _call_json(
        self,
        stage: str,
        system: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[dict[str, Any], LLMTrace]:
        self.ensure_available()
        started = time.monotonic()
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "vivatrace_result", "strict": True, "schema": schema},
            },
        }
        try:
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LocalLLMError(f"Локальная LLM не завершила этап «{stage}»: {error}") from error

        try:
            content = raw["choices"][0]["message"]["content"]
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise LocalLLMError(
                f"Локальная LLM вернула невалидный JSON на этапе «{stage}»."
            ) from error

        identity = self.identity()
        usage = raw.get("usage") or {}
        trace = LLMTrace(
            trace_id=str(raw.get("id") or f"local-{uuid4().hex}"),
            backend="llama.cpp",
            model=str(raw.get("model") or identity["model"]),
            model_sha256=str(identity["model_sha256"]),
            stage=stage,
            duration_ms=round((time.monotonic() - started) * 1000),
            created_at=datetime.now(UTC).isoformat(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        return result, trace

    def _is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=0.8) as response:  # noqa: S310
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False
