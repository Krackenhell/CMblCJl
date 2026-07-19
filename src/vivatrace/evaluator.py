from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Protocol

from .models import Evidence, ProbeQuestion


class AnswerEvaluator(Protocol):
    def evaluate(self, question: ProbeQuestion, answer: str) -> Evidence: ...


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


class RubricEvaluator:
    """Transparent local baseline for reproducible demos and comparison."""

    def evaluate(self, question: ProbeQuestion, answer: str) -> Evidence:
        normalized = _normalize(answer)
        hits = [
            any(_normalize(alias) in normalized for alias in concept_group)
            for concept_group in question.expected_concepts
        ]
        coverage = sum(hits) / max(len(hits), 1)

        misconception = None
        misconception_match_length = -1
        for label, patterns in question.misconception_patterns.items():
            for pattern in patterns:
                normalized_pattern = _normalize(pattern)
                if (
                    normalized_pattern in normalized
                    and len(normalized_pattern) > misconception_match_length
                ):
                    # Prefer the most specific phrase when one answer triggers
                    # both a general and a concrete misconception pattern.
                    misconception = label
                    misconception_match_length = len(normalized_pattern)

        length_factor = min(len(normalized.split()) / 35, 1.0)
        score = 0.85 * coverage + 0.15 * length_factor
        if misconception:
            score *= 0.55
        score = round(min(max(score, 0.0), 1.0), 3)
        confidence = round(0.55 + 0.35 * abs(score - 0.5) * 2, 3)

        covered = sum(hits)
        rationale = (
            f"В ответе раскрыто {covered} из {len(hits)} ожидаемых смысловых элементов."
        )
        if misconception:
            rationale += f" Обнаружен паттерн заблуждения: {misconception}."

        return Evidence(
            skill_id=question.skill_id,
            score=score,
            confidence=confidence,
            quote=answer.strip()[:360],
            rationale=rationale,
            misconception=misconception,
        )


class OpenAICompatibleEvaluator:
    """Optional LLM evaluator with a strict structured-output contract."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL") or None,
        )

    def evaluate(self, question: ProbeQuestion, answer: str) -> Evidence:
        schema = {
            "score": "float 0..1",
            "confidence": "float 0..1",
            "quote": "short exact evidence from student answer",
            "rationale": "brief explanation in Russian",
            "misconception": "string or null",
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты проверяешь предметное понимание, а не красноречие. "
                        "Не штрафуй за стиль, паузы или язык. Оцени только по рубрике. "
                        f"Верни JSON по схеме: {json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": asdict(question), "student_answer": answer},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return Evidence(
            skill_id=question.skill_id,
            score=float(payload["score"]),
            confidence=float(payload["confidence"]),
            quote=str(payload.get("quote") or answer[:360]),
            rationale=str(payload["rationale"]),
            misconception=payload.get("misconception"),
        )


def get_evaluator() -> tuple[AnswerEvaluator, str]:
    if os.getenv("LLM_API_KEY"):
        try:
            return OpenAICompatibleEvaluator(), "LLM + предметная рубрика"
        except Exception:
            pass
    return RubricEvaluator(), "Локальная воспроизводимая рубрика"
