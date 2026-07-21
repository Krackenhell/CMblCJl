from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .grading import grade_article_cloze, grade_structured_answer
from .models import Evidence, ProbeQuestion
from .rulebook import load_rulebook, rules_for_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m.gguf"
DEFAULT_FAST_MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_SERVER_PATH = PROJECT_ROOT / "tools" / "llama" / "llama-server.exe"
DEFAULT_VULKAN_SERVER_PATH = (
    PROJECT_ROOT / "tools" / "llama-vulkan" / "llama-server.exe"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "local-model-manifest.json"


def extract_surface_facts(answer: str) -> dict[str, Any]:
    """Extract literal evidence without making a pedagogical judgement."""
    normalized = " ".join(answer.lower().split())
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[а-яё]+", normalized)
    article_pairs = [
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
        if tokens[index] in {"a", "an", "the"}
    ]
    return {
        "normalized_answer": normalized,
        "first_token": tokens[0] if tokens else "",
        "starts_with_article": bool(tokens and tokens[0] in {"a", "an", "the"}),
        "article_noun_pairs": article_pairs,
    }


def sanitize_mixed_modal_negation(value: str) -> str:
    replacements = {
        "can": "cannot",
        "could": "could not",
        "must": "must not",
        "might": "might not",
        "should": "should not",
        "would": "would not",
    }
    for modal, replacement in replacements.items():
        value = re.sub(
            rf"\b{modal}\s+не\b",
            replacement,
            value,
            flags=re.IGNORECASE,
        )
    return value


def select_relevant_principle(rule: dict[str, Any], focus: str) -> str:
    summary_parts = [
        item.strip()
        for item in re.split(r"[.;]", str(rule.get("summary") or ""))
        if item.strip()
    ]
    principles = summary_parts + [str(item) for item in rule.get("principles") or []]
    if not principles:
        return str(rule.get("summary") or "проверяемое правило")
    focus_tokens = english_stems(focus)

    def overlap(principle: str) -> int:
        tokens = english_stems(principle)
        return len(focus_tokens & tokens)

    return max(principles, key=overlap)


def english_stems(value: str) -> set[str]:
    stems: set[str] = set()
    for token in re.findall(r"[a-z]+(?:'[a-z]+)?", value.lower()):
        if token in {"the", "a", "an", "i", "he", "she", "it", "we", "they", "some"}:
            continue
        if token.endswith("pped") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        stems.add(token)
    return stems


def generated_question_is_valid(
    item: dict[str, Any],
    required_form: str = "",
    *,
    transfer: bool = False,
    source_prompt: str = "",
) -> bool:
    text_key = "rule_focus" if transfer else "text"
    text = str(item.get(text_key) or "").strip()
    expected = str(item.get("expected_answer") or "").strip()
    if len(text) < 20 or len(expected) < 20:
        return False
    if not transfer and not text.endswith("?"):
        return False
    if not re.search(r"[А-Яа-яЁё]", text) or not re.search(
        r"[А-Яа-яЁё]", expected
    ):
        return False
    required_tokens = set(re.findall(r"[a-z]+(?:'[a-z]+)?", required_form.lower()))
    if required_tokens:
        generated_tokens = set(
            re.findall(r"[a-z]+(?:'[a-z]+)?", (text + " " + expected).lower())
        )
        if not required_tokens.issubset(generated_tokens):
            return False
    context_terms = english_stems(source_prompt) - english_stems(required_form)
    if context_terms:
        generated_context = english_stems(text + " " + expected)
        if not (context_terms & generated_context):
            return False
    return True


def grounded_transfer_example(
    source_prompt: str,
    required_form: str,
    principle: str,
    examples: list[str],
) -> str:
    stems = english_stems(source_prompt)
    normalized_form = required_form.lower().strip()
    lowered_principle = principle.lower()
    if "просьб" in lowered_principle or "please" in source_prompt.lower():
        return (
            "The manager asked the team to send the report. Здесь ask + object + "
            "to-infinitive передаёт просьбу в косвенной речи."
        )
    if "today" in lowered_principle and "that day" in lowered_principle:
        return (
            "Lena said that she was working that day. В прямой речи было today; "
            "при переносе точки отсчёта используется that day."
        )
    if "past perfect" in lowered_principle or "сдвиг времени" in lowered_principle:
        return (
            "Oleg said that he had finished the task. Past Simple из прямой речи "
            "сдвинут в Past Perfect после said."
        )
    if "if/whether" in lowered_principle or "косвен" in lowered_principle and "вопрос" in lowered_principle:
        return (
            "Mia asked whether I needed help. Общий косвенный вопрос вводится whether "
            "и использует прямой порядок слов."
        )
    if "non-defining" in lowered_principle or (
        "дополнительн" in lowered_principle and "запят" in lowered_principle
    ):
        return (
            "My phone, which is only a year old, still works well. Возраст — "
            "дополнительная информация об уже определённом my phone, поэтому нужны "
            "запятые и which, а не that."
        )
    if "where" in lowered_principle or "обстоятельство места" in lowered_principle:
        return (
            "The library where we usually study closes at nine. Where относится к "
            "месту и заменяет there в исходной мысли."
        )
    if "whose" in lowered_principle or "принадлежност" in lowered_principle:
        return (
            "I met a designer whose portfolio won an award. Whose показывает, что "
            "портфолио принадлежит дизайнеру."
        )
    if "относится к людям" in lowered_principle or "who/whom" in lowered_principle:
        return (
            "The teacher who helped me was very patient. Who относится к человеку — teacher."
        )
    if "относится к предмет" in lowered_principle or "which относится" in lowered_principle:
        return (
            "The book which I borrowed was useful. Which относится к предмету — book."
        )
    if "stop" in stems and normalized_form.startswith("to "):
        return (
            "The driver stopped to buy some water. Здесь stop + to-infinitive означает, "
            "что человек прервал другое действие ради покупки воды."
        )
    if "stop" in stems and normalized_form.endswith("ing"):
        return (
            "She stopped smoking last year. Здесь stop + -ing означает полностью "
            "прекратить действие."
        )
    if "remember" in stems and normalized_form.startswith("to "):
        return (
            "Remember to send the message. Здесь remember + to-infinitive означает "
            "не забыть выполнить действие."
        )
    if "remember" in stems and normalized_form.endswith("ing"):
        return (
            "I remember meeting her at university. Здесь remember + -ing относится к "
            "воспоминанию о прошлом действии."
        )
    example = examples[0] if examples else required_form
    return f"{example} Это новый пример правила: {principle}"


def _semantic_text(value: str) -> str:
    value = value.lower().replace("ё", "е").replace("’", "'")
    value = re.sub(r"[^a-zа-я0-9'+]+", " ", value)
    return " ".join(value.split())


def _semantic_term_matches(answer: str, term: str) -> bool:
    normalized_answer = f" {_semantic_text(answer)} "
    normalized_term = _semantic_text(term)
    if not normalized_term:
        return False
    if f" {normalized_term} " in normalized_answer:
        return True
    answer_tokens = normalized_answer.split()
    term_tokens = normalized_term.split()
    if len(term_tokens) == 1 and len(term_tokens[0]) >= 5:
        stem = term_tokens[0][: max(4, len(term_tokens[0]) - 2)]
        return any(token.startswith(stem) for token in answer_tokens)
    if len(term_tokens) > 1:
        for start in range(len(answer_tokens) - len(term_tokens) + 1):
            candidate = answer_tokens[start : start + len(term_tokens)]
            if all(
                actual == expected
                or (
                    len(actual) >= 5
                    and len(expected) >= 5
                    and actual[:4] == expected[:4]
                )
                for actual, expected in zip(candidate, term_tokens, strict=True)
            ):
                return True
    return False


def semantic_concept_coverage(
    expected_concepts: tuple[tuple[str, ...], ...] | list[list[str]], answer: str
) -> dict[str, Any]:
    groups = [tuple(str(term) for term in group) for group in expected_concepts if group]
    if not groups:
        return {"coverage": None, "covered": [], "missing": []}
    covered = [
        group[0]
        for group in groups
        if any(_semantic_term_matches(answer, term) for term in group)
    ]
    missing = [group[0] for group in groups if group[0] not in covered]
    return {
        "coverage": len(covered) / len(groups),
        "covered": covered,
        "missing": missing,
    }


def calibrated_viva_score(raw_score: float, concept_coverage: float | None) -> float:
    score = min(max(raw_score, 0.0), 1.0)
    if concept_coverage is None:
        return score
    # A prompt that explicitly asks for an explanation is not fully answered
    # when one of its required concepts is absent, even if the example is valid.
    if concept_coverage < 0.99:
        score = min(score, 0.70)
    ceilings = ((0.01, 0.30), (0.30, 0.50), (0.66, 0.70))
    for threshold, ceiling in ceilings:
        if concept_coverage < threshold:
            score = min(score, ceiling)
            break
    floors = ((0.99, 0.85), (0.66, 0.65), (0.49, 0.45), (0.30, 0.35))
    for threshold, floor in floors:
        if concept_coverage >= threshold:
            return max(score, floor)
    return score


def check_article_cloze(assignment: dict[str, Any], answer: str) -> dict[str, Any] | None:
    return grade_article_cloze(assignment, answer)


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
        self.fast_base_url = os.getenv(
            "LOCAL_LLM_FAST_URL", "http://127.0.0.1:8081"
        ).rstrip("/")
        self.model_path = Path(os.getenv("LOCAL_LLM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self.fast_model_path = Path(
            os.getenv("LOCAL_LLM_FAST_MODEL_PATH", str(DEFAULT_FAST_MODEL_PATH))
        )
        self.server_path = Path(os.getenv("LOCAL_LLM_SERVER_PATH", str(DEFAULT_SERVER_PATH)))
        self.vulkan_server_path = Path(
            os.getenv("LOCAL_LLM_VULKAN_SERVER_PATH", str(DEFAULT_VULKAN_SERVER_PATH))
        )
        self.manifest_path = Path(
            os.getenv("LOCAL_LLM_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH))
        )
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen2.5-7B-Instruct-Q4_K_M")
        self.fast_model_name = os.getenv(
            "LOCAL_LLM_FAST_MODEL", "Qwen2.5-3B-Instruct-Q4_K_M"
        )
        self.quality_mode = os.getenv("LOCAL_LLM_QUALITY_MODE", "0") == "1"
        self.rulebook = load_rulebook()

    def identity(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            # Windows PowerShell 5 writes UTF-8 JSON with a BOM by default.
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        running = self._is_running(self.base_url)
        fast_running = self._is_running(self.fast_base_url)
        runtime_exists = self.server_path.exists() or self.vulkan_server_path.exists()
        active_ready = (
            running or (self.model_path.exists() and runtime_exists)
            if self.quality_mode
            else fast_running or (self.fast_model_path.exists() and runtime_exists)
        )
        return {
            "ready": active_ready,
            "running": running,
            "fast_running": fast_running,
            "backend": "llama.cpp",
            "model": manifest.get("model", self.model_name),
            "model_sha256": manifest.get("model_sha256", "не вычислен"),
            "model_path": str(self.model_path),
            "fast_model": manifest.get("fast_model", self.fast_model_name),
            "fast_model_sha256": manifest.get("fast_model_sha256", "не вычислен"),
            "fast_model_path": str(self.fast_model_path),
            "quality_mode": self.quality_mode,
            "server_path": str(self.server_path),
            "accelerated_runtime": self.vulkan_server_path.exists(),
            "setup_command": r"powershell -ExecutionPolicy Bypass -File scripts\setup_local_llm.ps1",
        }

    def ensure_available(self) -> None:
        if self._is_running(self.base_url):
            return
        if not self.model_path.exists() or not (
            self.server_path.exists() or self.vulkan_server_path.exists()
        ):
            raise LocalLLMError(
                "Локальная LLM не установлена. Запустите scripts\\setup_local_llm.ps1."
            )
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        if self.vulkan_server_path.exists() and os.getenv("LOCAL_LLM_USE_VULKAN") == "1":
            accelerated_process = self._start_server(
                self.vulkan_server_path, accelerated=True
            )
            if self._wait_until_running(self.base_url, 75):
                return
            accelerated_process.terminate()
        self._start_server(self.server_path, accelerated=False)
        if self._wait_until_running(self.base_url, 120):
            return
        raise LocalLLMError("Локальная модель не запустилась. Проверьте logs/.")

    def ensure_fast_available(self) -> None:
        if self._is_running(self.fast_base_url):
            return
        if not self.fast_model_path.exists() or not self.server_path.exists():
            raise LocalLLMError(
                "Быстрая локальная LLM не установлена. Запустите scripts\\setup_local_llm.ps1."
            )
        self._start_fast_server()
        if self._wait_until_running(self.fast_base_url, 90):
            return
        raise LocalLLMError("Быстрая локальная модель не запустилась. Проверьте logs/.")

    def _start_server(self, server_path: Path, accelerated: bool) -> subprocess.Popen:
        log_dir = PROJECT_ROOT / "logs"
        suffix = "vulkan" if accelerated else "cpu"
        stdout = (log_dir / f"local-llm-{suffix}.stdout.log").open("a", encoding="utf-8")
        stderr = (log_dir / f"local-llm-{suffix}.stderr.log").open("a", encoding="utf-8")
        command = [
            str(server_path),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            self.base_url.rsplit(":", 1)[-1],
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
            "--threads",
            str(max((os.cpu_count() or 4) - 1, 2)),
            "--jinja",
            "--flash-attn",
            "auto",
            "--no-webui",
        ]
        if accelerated:
            command.extend(
                ["--gpu-layers", os.getenv("LOCAL_LLM_GPU_LAYERS", "18")]
            )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(  # noqa: S603
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )

    def _start_fast_server(self) -> subprocess.Popen:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout = (log_dir / "local-llm-fast.stdout.log").open("a", encoding="utf-8")
        stderr = (log_dir / "local-llm-fast.stderr.log").open("a", encoding="utf-8")
        command = [
            str(self.server_path),
            "--model",
            str(self.fast_model_path),
            "--host",
            "127.0.0.1",
            "--port",
            self.fast_base_url.rsplit(":", 1)[-1],
            "--ctx-size",
            "3072",
            "--parallel",
            "1",
            "--threads",
            str(max((os.cpu_count() or 4) - 1, 2)),
            "--jinja",
            "--flash-attn",
            "auto",
            "--no-webui",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(  # noqa: S603
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )

    def _wait_until_running(self, base_url: str, timeout_seconds: int) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._is_running(base_url):
                return True
            time.sleep(1)
        return False

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
                "criterion_results",
            ],
            "properties": {
                "submission_score": {"type": "number", "minimum": 0, "maximum": 1},
                "is_correct": {"type": "boolean"},
                "feedback": {"type": "string", "maxLength": 220},
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
                            "diagnosis": {"type": "string", "maxLength": 200},
                        },
                    },
                },
                "criterion_results": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "criterion",
                            "status",
                            "student_evidence",
                            "issue",
                            "correction",
                        ],
                        "properties": {
                            "criterion": {"type": "string", "maxLength": 160},
                            "status": {
                                "type": "string",
                                "enum": ["correct", "partial", "incorrect"],
                            },
                            "student_evidence": {"type": "string", "maxLength": 180},
                            "issue": {"type": "string", "maxLength": 220},
                            "correction": {"type": "string", "maxLength": 180},
                        },
                    },
                },
            },
        }
        rubric = assignment.get("rubric") or {}
        grounded_rules = rules_for_assignment(assignment, self.rulebook)
        objective_check = grade_structured_answer(assignment, answer)
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
            "criterion_results — это конкретный разбор каждого пункта рубрики: процитируй относящийся "
            "к нему фрагмент студента, назови точную ошибку и покажи исправленный вариант. Не используй "
            "одинаковую общую формулировку для разных ответов. Незначительная опечатка вне проверяемого "
            "правила не является грамматической ошибкой и не должна заметно снижать балл. Каждое поле "
            "criterion_results — не длиннее одного короткого предложения. "
            "Если решение правильное, mode=viva. Если решение неправильное, mode=diagnostic. "
            "Пиши объяснения по-русски, а примеры "
            "английского оставляй на английском. Верни только JSON по заданной схеме."
        )
        grade_trace = None
        if objective_check:
            submission_score = float(objective_check["score"])
            is_correct = bool(objective_check["correct"])
            wrong_positions = [
                str(slot["position"])
                for slot in objective_check["slots"]
                if not slot["correct"]
            ]
            unit = "позициях" if "eng_articles" in assignment["skill_ids"] else "пунктах"
            feedback = (
                "Все ответы совпадают с проверяемым ключом."
                if is_correct
                else (
                    f'Работа выполнена на {submission_score:.0%}. Нужно доработать отдельные '
                    f'компоненты в {unit}: {", ".join(wrong_positions)}.'
                )
            )
            result = {
                "submission_score": submission_score,
                "is_correct": is_correct,
                "feedback": feedback,
                "mode": "viva" if is_correct else "diagnostic",
                "skill_results": [
                    {
                        "skill_id": skill_id,
                        "score": submission_score,
                        "diagnosis": feedback,
                    }
                    for skill_id in assignment["skill_ids"]
                ],
                "criterion_results": [
                    {
                        "criterion": f'Пункт {slot["position"]}',
                        "status": (
                            "correct"
                            if slot["correct"]
                            else "partial"
                            if float(slot.get("score", 0)) >= 0.4
                            else "incorrect"
                        ),
                        "score": float(slot.get("score", int(slot["correct"]))),
                        "student_evidence": slot["student_evidence"],
                        "issue": slot["issue"],
                        "correction": slot["expected_phrase"],
                    }
                    for slot in objective_check["slots"]
                ],
                "objective_check": objective_check,
            }
        else:
            user_payload = {
                "subject": assignment.get("subject"),
                "topic": assignment["topic"],
                "task": assignment["instructions"],
                "rubric": rubric,
                "grounded_rules": grounded_rules,
                "skills": {
                    skill_id: skill_names[skill_id] for skill_id in assignment["skill_ids"]
                },
                "student_answer": answer,
            }
            result, grade_trace = self._call_json(
                "проверка открытого задания",
                system,
                user_payload,
                grade_schema,
                650,
                fast=not self.quality_mode,
            )
            submission_score = float(result["submission_score"])
            is_correct = bool(result["is_correct"]) and submission_score >= 0.75
            result["is_correct"] = is_correct
            result["mode"] = "viva" if is_correct else "diagnostic"
            result["objective_check"] = None
            if is_correct:
                result["submission_score"] = max(submission_score, 0.85)
                for skill_result in result["skill_results"]:
                    skill_result["score"] = max(float(skill_result["score"]), 0.85)
            else:
                result["submission_score"] = min(submission_score, 0.7)
        question_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["first_question", "transfer_question"],
            "properties": {
                "first_question": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["skill_id", "text", "expected_answer"],
                    "properties": {
                        "skill_id": {"type": "string", "enum": assignment["skill_ids"]},
                        "text": {"type": "string", "maxLength": 260},
                        "expected_answer": {"type": "string", "maxLength": 260},
                    },
                },
                "transfer_question": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["skill_id", "rule_focus", "expected_answer"],
                    "properties": {
                        "skill_id": {"type": "string", "enum": assignment["skill_ids"]},
                        "rule_focus": {"type": "string", "maxLength": 180},
                        "expected_answer": {"type": "string", "maxLength": 300},
                    },
                },
            },
        }
        common_question_system = (
            "Создай сразу ДВА коротких самодостаточных Viva-вопроса по одному указанному правилу. "
            "Первый вопрос проверяет понимание конкретного source_prompt, второй — перенос того же правила "
            "в новый контекст. Не смешивай слова или ситуации из разных пунктов задания. Не утверждай, что "
            "to-infinitive ставится после смыслового глагола из скобок: управляющая конструкция дана в "
            "source_prompt. Поля пиши по-русски, английским оставляй только примеры и формы. Первый text "
            "должен оканчиваться знаком вопроса. В rule_focus сформулируй понятное условие применения, а не "
            "обрывок правила. В каждом expected_answer дай прямой содержательный ответ, не повтор вопроса. "
            "Верни только JSON указанной схемы."
        )
        wrong_slots = [
            slot
            for slot in (objective_check or {}).get("slots", [])
            if not slot.get("correct")
        ]
        all_slots = (objective_check or {}).get("slots", [])
        focused_slot = wrong_slots[0] if wrong_slots else (all_slots[0] if all_slots else {})
        required_form = str(focused_slot.get("expected_phrase") or "")
        source_prompt = str(focused_slot.get("prompt") or assignment["instructions"])
        grounded_focus = required_form
        rule = self.rulebook.get(assignment["skill_ids"][0], {})
        diagnostic = dict(focused_slot.get("diagnostic") or {})
        verified_focus = diagnostic or dict(focused_slot.get("probe") or {})
        principle = str(verified_focus.get("rule_focus") or select_relevant_principle(
            rule, f"{source_prompt} {grounded_focus}"
        ))
        question_payload = {
            "topic": assignment["topic"],
            "student_answer": answer,
            "mode": result["mode"],
            "criterion_results": result["criterion_results"],
            "grounded_rules": grounded_rules,
            "source_prompt": source_prompt,
            "required_form": required_form,
            "grounded_rule_focus": principle,
            "verified_diagnostic": verified_focus,
            "instructions": (
                "Если mode=diagnostic, первый вопрос просит объяснить исправление именно source_prompt. "
                "Если mode=viva, он просит объяснить уже правильную форму. Второй вопрос просит придумать "
                "новое полное английское предложение и объяснить смысл формы."
            ),
        }
        question_pair, question_trace = self._call_json(
            "формирование двух вопросов",
            common_question_system,
            question_payload,
            question_schema,
            520,
            fast=True,
        )
        question_traces = [question_trace]
        first_item = dict(question_pair.get("first_question") or {})
        second_item = dict(question_pair.get("transfer_question") or {})
        examples = [str(item) for item in rule.get("examples") or []]
        if verified_focus:
            # The local model may paraphrase, but it may not change what is being tested.
            # The verified subject contract therefore owns the concrete question and answer.
            first_item = {
                "skill_id": assignment["skill_ids"][0],
                "text": str(verified_focus["question"]),
                "expected_answer": str(verified_focus["expected_answer"]),
            }
        elif not generated_question_is_valid(
            first_item, required_form, source_prompt=source_prompt
        ):
            if diagnostic:
                question_text = str(diagnostic["question"])
            elif wrong_slots:
                position = wrong_slots[0].get("position")
                question_text = (
                    f"В пункте {position} дано «{source_prompt}». Почему здесь нужна форма "
                    f"«{grounded_focus}» и какой смысл она выражает?"
                )
            else:
                question_text = (
                    f"В пункте дано «{source_prompt}». Почему форма «{grounded_focus}» "
                    "соответствует смыслу предложения?"
                )
            first_item = {
                "skill_id": assignment["skill_ids"][0],
                "text": question_text,
                "expected_answer": str(
                    verified_focus.get("expected_answer") or f"{grounded_focus}: {principle}"
                ),
            }
        if not generated_question_is_valid(
            second_item,
            "",
            transfer=True,
            source_prompt=principle,
        ):
            second_item = {
                "skill_id": assignment["skill_ids"][0],
                "rule_focus": principle,
                "expected_answer": grounded_transfer_example(
                    source_prompt, grounded_focus, principle, examples
                ),
            }
        rule_focus = second_item.pop("rule_focus").rstrip(".?")
        second_item["text"] = (
            f"Составьте одно новое английское предложение, где действует правило «{rule_focus}», "
            "и кратко объясните выбор формы?"
        )
        for item in (first_item, second_item):
            for key, value in item.items():
                if isinstance(value, str):
                    item[key] = sanitize_mixed_modal_negation(value)
        question_items = [first_item, second_item]
        question_item_traces = [question_trace, question_trace]
        result["questions"] = [
            ProbeQuestion(
                id=f"local-{trace.trace_id}-{index}",
                skill_id=item["skill_id"],
                text=item["text"],
                purpose=(
                    "Подтвердить самостоятельное понимание и перенос правила."
                    if result["mode"] == "viva"
                    else "Точно определить пробел и помочь восстановить правило."
                ),
                expected_concepts=tuple(
                    tuple(str(term) for term in group)
                    for group in verified_focus.get("expected_concepts", [])
                ),
                rule_id=item["skill_id"],
                expected_answer=item["expected_answer"],
                context_constraint="Нет дополнительных ограничений контекста.",
            )
            for index, (item, trace) in enumerate(
                zip(question_items, question_item_traces, strict=True), start=1
            )
        ]
        traces = ([grade_trace] if grade_trace is not None else []) + question_traces
        return result, traces

    def evaluate_answer(
        self,
        assignment: dict[str, Any],
        question: ProbeQuestion,
        answer: str,
    ) -> tuple[Evidence, LLMTrace]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "score",
                "confidence",
                "verdict",
                "what_was_correct",
                "what_needs_improvement",
                "correct_answer",
                "typo_handling",
            ],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "partial", "incorrect"],
                },
                "what_was_correct": {"type": "string"},
                "what_needs_improvement": {"type": "string"},
                "correct_answer": {"type": "string"},
                "typo_handling": {"type": "string"},
            },
        }
        system = (
            "Ты строгий оценщик ответа на проверочный вопрос. Оцени только релевантное предметное "
            "понимание. Совпадение отдельных слов, перефразирование вопроса, фразы «не знаю», "
            "«вся информация», «нужно воспроизвести» и случайный текст не являются доказательством "
            "знания и должны получать score 0–0.10. Частично верная причинная связь — 0.3–0.6; "
            "полный корректный ответ — 0.75–1.0. Если проверяемое правило раскрыто и пример валиден, "
            "ставь 0.85–1.0; стилистические пожелания вроде «ответить формальнее» не снижают балл. "
            "Оценивай только то, что явно спросил вопрос. Не придумывай тематические ограничения: если "
            "context_constraint говорит, что ограничений нет, прими любой грамматически и смыслово валидный "
            "пример, включая пример про home вместо travel. Если студент уже привёл требуемый пример, нельзя "
            "писать, что примера нет. Незначительные опечатки вроде 'tooka a' вместо 'took a' не снижают "
            "балл за понимание артиклей, если целевая конструкция ясна; опиши это в typo_handling. "
            "what_was_correct, what_needs_improvement и correct_answer должны быть конкретными и короткими. "
            "Перед выводом проверь буквальный текст: нельзя утверждать, что артикль или пример отсутствует, "
            "если он есть в student_answer. surface_facts вычислены кодом из буквального ответа и являются "
            "авторитетными для наличия слов: если first_token=home и starts_with_article=false, перед home "
            "нулевой артикль; если article_noun_pairs содержит и 'a train', и 'the train', обе формы присутствуют. "
            "Калибровка: (1) на просьбу дать любой zero-article example ответ "
            "'Home is a place where I feel comfortable' корректен и получает >=0.85; (2) ответ с 'I tooka a "
            "train ... I enjoyed the train' правильно показывает a при первом и the при повторном упоминании, "
            "а tooka — лишь опечатка, итог >=0.85; (3) 'travel is one of the greatest human passions' — "
            "валидный пример general concept, итог >=0.85. Для correct напиши в what_needs_improvement, "
            "что по проверяемому правилу исправления не нужны. Все поля пиши по-русски; не используй "
            "китайский или другой язык, кроме английских примеров из задания. typo_handling оставь "
            "пустым, если нет явной орфографической опечатки: альтернативная валидная формулировка вроде "
            "'is a place' вместо 'is where' не является опечаткой. "
            "Если question.expected_concepts не пуст, оцени долю явно раскрытых атомарных понятий. Не требуй "
            "дословного совпадения с expected_answer и не штрафуй правильное перефразирование. Итоговый score "
            "должен отражать покрытие понятий и наличие противоречий. "
            "Верни только JSON."
        )
        payload = {
            "grounded_rule": {
                key: value
                for key, value in self.rulebook.get(
                    question.rule_id or question.skill_id, {}
                ).items()
                if key in {"title", "summary", "principles"}
            },
            "question": asdict(question),
            "student_answer": answer,
            "surface_facts": extract_surface_facts(answer),
        }
        result, trace = self._call_json(
            "оценка ответа viva", system, payload, schema, 230, fast=True
        )
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = sanitize_mixed_modal_negation(value)
        coverage = semantic_concept_coverage(question.expected_concepts, answer)
        result["score"] = calibrated_viva_score(
            float(result["score"]), coverage["coverage"]
        )
        if coverage["covered"] and str(result["verdict"]) != "correct":
            result["what_was_correct"] = (
                "В ответе подтверждены элементы: " + ", ".join(coverage["covered"]) + "."
            )
        if coverage["missing"]:
            result["what_needs_improvement"] = (
                "Нужно явно раскрыть: " + ", ".join(coverage["missing"]) + "."
            )
        score = float(result["score"])
        verdict = "correct" if score >= 0.75 else "partial" if score >= 0.3 else "incorrect"
        result["verdict"] = verdict
        if verdict == "correct":
            result["score"] = max(float(result["score"]), 0.85)
            result["what_was_correct"] = (
                f"Ответ применяет проверяемое правило: «{answer.strip()}»."
            )
            result["what_needs_improvement"] = (
                "По проверяемому правилу исправления не нужны."
            )
            result["correct_answer"] = answer.strip()
        elif verdict == "partial":
            result["score"] = min(max(float(result["score"]), 0.3), 0.7)
            result["correct_answer"] = question.expected_answer or result["correct_answer"]
        else:
            result["score"] = min(float(result["score"]), 0.3)
            result["correct_answer"] = question.expected_answer or result["correct_answer"]
        rule = self.rulebook.get(question.rule_id or question.skill_id, {})
        rationale = (
            f'Верно: {result["what_was_correct"]} '
            f'Нужно улучшить: {result["what_needs_improvement"]}'
        )
        misconception = (
            None if verdict == "correct" else str(result["what_needs_improvement"])
        )
        evidence = Evidence(
            skill_id=question.skill_id,
            score=round(float(result["score"]), 3),
            confidence=round(float(result["confidence"]), 3),
            quote=answer.strip()[:360],
            rationale=rationale,
            misconception=misconception,
            source="local_llm",
            evaluator_model=trace.model,
            trace_id=trace.trace_id,
            question_text=question.text,
            question_purpose=question.purpose,
            rule_id=question.rule_id or question.skill_id,
            rule_title=rule.get("title"),
            rule_url=rule.get("source_url"),
            verdict=verdict,
            what_was_correct=str(result["what_was_correct"]),
            what_needs_improvement=str(result["what_needs_improvement"]),
            correct_answer=str(result["correct_answer"]),
            typo_handling=str(result.get("typo_handling") or ""),
        )
        return evidence, trace

    def finalize_learning(
        self,
        assignment: dict[str, Any],
        assessment: dict[str, Any],
        evidence: list[Evidence],
        cohort_context: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[LLMTrace]]:
        student_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["branch", "student_activity"],
            "properties": {
                "branch": {"type": "string", "enum": ["transfer", "remediation"]},
                "student_activity": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "title",
                        "why",
                        "explanation",
                        "worked_example",
                        "practice_task",
                        "success_criteria",
                    ],
                    "properties": {
                        "title": {"type": "string", "maxLength": 100},
                        "why": {"type": "string", "maxLength": 180},
                        "explanation": {"type": "string", "maxLength": 320},
                        "worked_example": {"type": "string", "maxLength": 320},
                        "practice_task": {"type": "string", "maxLength": 260},
                        "success_criteria": {"type": "string", "maxLength": 220},
                    },
                },
            },
        }
        teacher_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["focus_topic", "reason", "lesson_plan", "evidence_summary"],
            "properties": {
                "focus_topic": {"type": "string", "maxLength": 120},
                "reason": {"type": "string", "maxLength": 240},
                "lesson_plan": {"type": "string", "maxLength": 320},
                "evidence_summary": {"type": "string", "maxLength": 260},
            },
        }
        student_system = (
            "Ты методист и создаёшь персональный следующий шаг только по evidence и grounded_rules. "
            "Пиши по-русски, английским оставляй только учебные примеры. Если исходное решение неверно "
            "или хотя бы один viva score ниже 0.75, выбери remediation; иначе transfer. Для remediation "
            "назови точный пробел, объясни конкретное правило, дай ОДИН новый правильный разобранный пример "
            "и ОДНО новое короткое задание без готового ответа. Для transfer дай новое более сложное задание "
            "на то же правило. Не копируй исходное упражнение или вопрос viva. Нулевой артикль означает "
            "отсутствие a/an/the: не ставь символ тире перед словом в обычном английском предложении. "
            "worked_example обязан быть грамматически верным, practice_task — однозначным. Каждый блок — "
            "не более двух коротких предложений. Верни JSON."
        )
        compact_evidence = [
            {
                "question": item.question_text,
                "answer": item.quote,
                "score": item.score,
                "gap": item.what_needs_improvement,
                "correct_answer": item.correct_answer,
                "rule_id": item.rule_id,
            }
            for item in evidence
        ]
        student_payload = {
            "topic": assignment["topic"],
            "submission_correct": assessment["is_correct"],
            "evidence": compact_evidence,
            "grounded_rules": rules_for_assignment(assignment, self.rulebook),
        }
        student_result, student_trace = self._call_json(
            "персональный следующий шаг",
            student_system,
            student_payload,
            student_schema,
            460,
            fast=not self.quality_mode,
        )
        for key, value in student_result.get("student_activity", {}).items():
            if isinstance(value, str):
                student_result["student_activity"][key] = sanitize_mixed_modal_negation(value)
        needs_remediation = not bool(assessment["is_correct"]) or any(
            item.score < 0.75 for item in evidence
        )
        student_result["branch"] = "remediation" if needs_remediation else "transfer"
        if needs_remediation and evidence:
            weakest = min(evidence, key=lambda item: item.score)
            rule = self.rulebook.get(weakest.rule_id or weakest.skill_id, {})
            if rule:
                examples = list(rule.get("examples") or [])
                wrong_slots = [
                    slot
                    for slot in (assessment.get("objective_check") or {}).get("slots", [])
                    if not slot.get("correct")
                ]
                focus_text = " ".join(
                    str(slot.get("expected_phrase") or "") for slot in wrong_slots
                ) or weakest.correct_answer
                principle = select_relevant_principle(rule, focus_text)
                example_text = " · ".join(examples[:2])
                if wrong_slots:
                    why = " ".join(
                        (
                            f'Пункт {slot.get("position")}: '
                            f'«{slot.get("student_evidence") or "ответ отсутствует"}» '
                            f'→ «{slot.get("expected_phrase")}».'
                        )
                        for slot in wrong_slots[:2]
                    )
                else:
                    why = (
                        f'Ответ на вопрос «{weakest.question_text}» получил {weakest.score:.0%}; '
                        "правило нужно подтвердить ещё раз."
                    )
                student_result["student_activity"] = {
                    "title": f'Разбор: {rule["title"]}',
                    "instructions": (
                        "Изучите правило и примеры, затем выполните повторную практику."
                    ),
                    "why": why,
                    "explanation": str(rule["summary"]),
                    "worked_example": f"Сравните примеры: {example_text}",
                    "practice_task": (
                        f"Составьте два новых английских предложения по правилу «{principle}» "
                        "и кратко объясните выбор формы."
                    ),
                    "success_criteria": (
                        f"В обоих предложениях соблюдено правило: {principle}"
                    ),
                }
        else:
            student_result["student_activity"]["instructions"] = (
                "Изучите объяснение и пример, затем выполните новое задание."
            )
        teacher_system = (
            "Ты методист преподавателя. По фактическим данным cohort_results и current_evidence предложи "
            "один конкретный фокус следующего занятия и короткий план из трёх действий. Пиши по-русски. "
            "Используй только rule_id, score и objective_errors: свободные диагнозы не передаются намеренно. "
            "Не пиши, что одна и та же форма должна быть заменена сама на себя. lesson_plan обязан содержать "
            "ровно три законченных действия с метками 1), 2), 3) и закончиться точкой. Не придумывай студентов, "
            "ответы или проценты. Не давай общих советов без связи с evidence. Верни JSON."
        )
        objective_errors = [
            {
                "position": slot.get("position"),
                "student_form": slot.get("student_evidence"),
                "expected_form": slot.get("expected_phrase"),
            }
            for slot in (assessment.get("objective_check") or {}).get("slots", [])
            if not slot.get("correct")
        ]
        teacher_payload = {
            "topic": assignment["topic"],
            "cohort_results": cohort_context,
            "current_evidence": [
                {
                    "question": item.question_text,
                    "score": item.score,
                    "rule_id": item.rule_id,
                }
                for item in evidence
            ],
            "objective_errors": objective_errors,
        }
        teacher_result, teacher_trace = self._call_json(
            "рекомендация преподавателю",
            teacher_system,
            teacher_payload,
            teacher_schema,
            300,
            fast=not self.quality_mode,
        )
        for key, value in teacher_result.items():
            if isinstance(value, str):
                teacher_result[key] = sanitize_mixed_modal_negation(value)
        return {
            **student_result,
            "teacher_recommendation": teacher_result,
        }, [student_trace, teacher_trace]

    def _call_json(
        self,
        stage: str,
        system: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        max_tokens: int,
        fast: bool = False,
        retry_on_invalid: bool = True,
    ) -> tuple[dict[str, Any], LLMTrace]:
        if fast:
            self.ensure_fast_available()
        else:
            self.ensure_available()
        base_url = self.fast_base_url if fast else self.base_url
        model_name = self.fast_model_name if fast else self.model_name
        started = time.monotonic()
        request_body = {
            "model": model_name,
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
                f"{base_url}/v1/chat/completions",
                data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LocalLLMError(f"Локальная LLM не завершила этап «{stage}»: {error}") from error

        try:
            content = raw["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end >= start:
                content = content[start : end + 1]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            if retry_on_invalid:
                retry_system = (
                    system
                    + " Предыдущий ответ не разобрался как JSON. Повтори результат короче, "
                    "строго одним полным JSON-объектом без Markdown и текста вокруг."
                )
                return self._call_json(
                    stage,
                    retry_system,
                    payload,
                    schema,
                    max(max_tokens * 2, 460),
                    fast=fast,
                    retry_on_invalid=False,
                )
            raise LocalLLMError(
                f"Локальная LLM вернула невалидный JSON на этапе «{stage}»."
            ) from error

        identity = self.identity()
        model_hash = (
            identity["fast_model_sha256"] if fast else identity["model_sha256"]
        )
        usage = raw.get("usage") or {}
        trace = LLMTrace(
            trace_id=str(raw.get("id") or f"local-{uuid4().hex}"),
            backend="llama.cpp",
            model=str(raw.get("model") or identity["model"]),
            model_sha256=str(model_hash),
            stage=stage,
            duration_ms=round((time.monotonic() - started) * 1000),
            created_at=datetime.now(UTC).isoformat(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        return result, trace

    def _is_running(self, base_url: str | None = None) -> bool:
        base_url = base_url or self.base_url
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=0.8) as response:  # noqa: S310
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False
