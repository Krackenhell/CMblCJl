from __future__ import annotations

import re
from itertools import product
from typing import Any


NUMBERED_ITEM_RE = re.compile(
    r"(?<!\w)(\d+)\s*\)\s*(.*?)(?=(?<!\w)\d+\s*\)|\Z)",
    re.DOTALL,
)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def normalize_answer(value: str) -> str:
    normalized = value.lower().replace("’", "'").replace("`", "'")
    contractions = {
        "can't": "cannot",
        "couldn't": "could not",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "wasn't": "was not",
        "weren't": "were not",
        "wouldn't": "would not",
        "won't": "will not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
    }
    for contraction, expanded in contractions.items():
        normalized = normalized.replace(contraction, expanded)
    normalized = re.sub(r"\bcannot\b", "can not", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def parse_numbered_items(value: str) -> list[str]:
    matches = list(NUMBERED_ITEM_RE.finditer(value.strip()))
    if not matches:
        return []
    indexes = [int(match.group(1)) for match in matches]
    if indexes != list(range(1, len(indexes) + 1)):
        return []
    return [match.group(2).strip().strip(";. ") for match in matches]


def accepted_variants(reference: str) -> list[str]:
    variants = [reference]
    replacements = (
        ("might/could have", ("might have", "could have")),
        ("might/could", ("might", "could")),
        ("whether/if", ("whether", "if")),
        ("learned/learnt", ("learned", "learnt")),
        ("where/in which", ("where", "in which")),
    )
    choices: list[tuple[str, tuple[str, ...]]] = [
        (source, options) for source, options in replacements if source.lower() in reference.lower()
    ]
    if choices:
        variants = []
        for selected in product(*(options for _, options in choices)):
            candidate = reference
            for (source, _), replacement in zip(choices, selected, strict=True):
                candidate = re.sub(re.escape(source), replacement, candidate, flags=re.IGNORECASE)
            variants.append(candidate)
    expanded = list(variants)
    for variant in variants:
        expanded.append(re.sub(r"\bsaid that\b", "said", variant, flags=re.IGNORECASE))
    return list(dict.fromkeys(expanded))


def answer_matches(actual: str, expected_variants: list[str]) -> bool:
    normalized_actual = normalize_answer(actual)
    if not normalized_actual:
        return False
    for expected in expected_variants:
        normalized_expected = normalize_answer(expected)
        if normalized_actual == normalized_expected:
            return True
        trailing_sections = (
            " because ",
            " explanation ",
            " explanations ",
            " meaning difference ",
            " which clause ",
        )
        if any(
            normalized_actual.startswith(normalized_expected + marker)
            for marker in trailing_sections
        ):
            return True
    return False


def grade_numbered_answer(assignment: dict[str, Any], answer: str) -> dict[str, Any] | None:
    reference = str((assignment.get("rubric") or {}).get("reference_answer") or "")
    reference_items = parse_numbered_items(reference)
    if len(reference_items) < 2:
        return None
    answer_items = parse_numbered_items(answer)
    slots = []
    for index, expected in enumerate(reference_items, start=1):
        actual = answer_items[index - 1] if index <= len(answer_items) else ""
        variants = accepted_variants(expected)
        correct = answer_matches(actual, variants)
        slots.append(
            {
                "position": index,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "student_evidence": actual or "Ответ отсутствует",
                "expected_phrase": " / ".join(variants[:3]),
                "issue": (
                    ""
                    if correct
                    else (
                        f"Пункт {index} не заполнен."
                        if not actual
                        else f"В пункте {index} получено «{actual}», ожидается «{expected}»."
                    )
                ),
            }
        )
    correct_count = sum(bool(slot["correct"]) for slot in slots)
    return {
        "source": "deterministic_answer_key",
        "correct": correct_count == len(slots),
        "score": correct_count / len(slots),
        "slots": slots,
    }


def grade_article_cloze(assignment: dict[str, Any], answer: str) -> dict[str, Any] | None:
    if "eng_articles" not in assignment.get("skill_ids", []):
        return None
    instructions = str(assignment.get("instructions") or "")
    reference = str((assignment.get("rubric") or {}).get("reference_answer") or "")
    if "___" not in instructions or not reference:
        return None

    segments = [part.strip(" .‘’'\"") for part in reference.split(";") if part.strip()]
    if len(segments) != instructions.count("___"):
        return None

    answer_tokens = TOKEN_RE.findall(answer)
    lowered_answer = [token.lower() for token in answer_tokens]
    instruction_sections = instructions.split("___")
    articles = {"a", "an", "the"}
    cursor = 0
    slots: list[dict[str, Any]] = []

    for index, segment in enumerate(segments, start=1):
        reference_tokens = TOKEN_RE.findall(segment)
        if not reference_tokens:
            return None
        first = reference_tokens[0].lower()
        expected = first if first in articles else "—"
        anchor = reference_tokens[1:] if first in articles else reference_tokens
        anchor_lower = [token.lower() for token in anchor]
        match_index = next(
            (
                position
                for position in range(cursor, len(lowered_answer) - len(anchor_lower) + 1)
                if lowered_answer[position : position + len(anchor_lower)] == anchor_lower
            ),
            None,
        )
        left_context = TOKEN_RE.findall(instruction_sections[index - 1])[-3:]
        left_context_lower = [token.lower() for token in left_context]
        left_match = None
        if match_index is not None and left_context_lower:
            left_match = next(
                (
                    position
                    for position in range(match_index - len(left_context_lower), -1, -1)
                    if lowered_answer[position : position + len(left_context_lower)]
                    == left_context_lower
                ),
                None,
            )
        if match_index is None:
            actual = "не найдено"
            evidence = "Фрагмент отсутствует в ответе"
            cursor = len(lowered_answer)
        elif left_match is not None:
            gap_start = left_match + len(left_context_lower)
            inserted_tokens = answer_tokens[gap_start:match_index]
            actual = " ".join(token.lower() for token in inserted_tokens) if inserted_tokens else "—"
            evidence = " ".join(inserted_tokens + answer_tokens[match_index : match_index + len(anchor)])
            cursor = match_index + len(anchor)
        else:
            previous = lowered_answer[match_index - 1] if match_index > 0 else ""
            actual = previous if previous in articles else "—"
            phrase_start = match_index - 1 if previous in articles else match_index
            evidence = " ".join(answer_tokens[phrase_start : match_index + len(anchor)])
            cursor = match_index + len(anchor)
        correct = actual == expected
        expected_phrase = " ".join(([expected] if expected != "—" else []) + anchor)
        actual_label = "нулевой артикль" if actual == "—" else actual
        expected_label = "нулевой артикль" if expected == "—" else expected
        slots.append(
            {
                "position": index,
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "student_evidence": evidence,
                "expected_phrase": expected_phrase,
                "issue": (
                    ""
                    if correct
                    else f"Получено «{actual_label}», но по эталону нужен {expected_label}."
                ),
            }
        )

    correct_count = sum(bool(slot["correct"]) for slot in slots)
    return {
        "source": "deterministic_answer_key",
        "correct": correct_count == len(slots),
        "score": correct_count / len(slots),
        "slots": slots,
    }


def grade_structured_answer(
    assignment: dict[str, Any], answer: str
) -> dict[str, Any] | None:
    result = grade_article_cloze(assignment, answer) or grade_numbered_answer(
        assignment, answer
    )
    if result:
        result["rule_id"] = str((assignment.get("skill_ids") or ["general"])[0])
    return result
