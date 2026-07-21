from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import product
from typing import Any


NUMBERED_ITEM_RE = re.compile(
    r"(?<!\w)(\d+)\s*\)\s*(.*?)(?=(?<!\w)\d+\s*\)|\Z)",
    re.DOTALL,
)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CONTENT_STOP_WORDS = {
    "a", "an", "the", "that", "this", "to", "from", "in", "on", "at", "of",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "said", "asked", "told", "if", "whether", "was", "were", "had", "has", "have",
}


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
    zero_markers = {"—", "–", "-", "zero", "zero article", "no article"}
    if actual.strip().lower() in zero_markers and any(
        expected.strip().lower() in zero_markers for expected in expected_variants
    ):
        return True
    normalized_actual = normalize_answer(actual)
    if not normalized_actual:
        return False
    for expected in expected_variants:
        normalized_expected = normalize_answer(expected)
        if normalized_actual == normalized_expected:
            return True
        trailing_sections = {
            "because",
            "explanation",
            "explanations",
            "meaning difference",
            "which clause",
        }
        remainder = normalized_actual[len(normalized_expected) :].strip()
        if any(
            remainder == marker or remainder.startswith(marker + " ")
            for marker in trailing_sections
        ):
            return True
    return False


def _tokens(value: str) -> list[str]:
    return normalize_answer(value).split()


def _contains_tokens(value: str, phrase: str) -> bool:
    haystack = f" {' '.join(_tokens(value))} "
    needle = f" {' '.join(_tokens(phrase))} "
    return bool(needle.strip()) and needle in haystack


def _component(
    code: str,
    label: str,
    weight: float,
    score: float,
    rule_focus: str,
    question: str,
    expected_answer: str,
    expected_concepts: list[list[str]],
) -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "weight": weight,
        "score": min(max(score, 0.0), 1.0),
        "rule_focus": rule_focus,
        "question": question,
        "expected_answer": expected_answer,
        "expected_concepts": expected_concepts,
    }


def grade_reported_speech_slot(prompt: str, actual: str, expected: str) -> dict[str, Any]:
    """Score reusable reported-speech transformations by atomic grammar components."""
    actual_tokens = _tokens(actual)
    expected_tokens = _tokens(expected)
    if not actual_tokens:
        return {"score": 0.0, "components": [], "primary_error": None}

    components: list[dict[str, Any]] = []
    content_expected = {token for token in expected_tokens if token not in CONTENT_STOP_WORDS}
    content_actual = {token for token in actual_tokens if token not in CONTENT_STOP_WORDS}
    content_score = (
        len(content_expected & content_actual) / len(content_expected)
        if content_expected
        else 1.0
    )
    components.append(
        _component(
            "content",
            "содержание исходного сообщения",
            0.25,
            content_score,
            "Содержание прямой речи должно сохраняться при преобразовании.",
            f"Какая основная информация из «{prompt}» должна сохраниться в косвенной речи?",
            expected,
            [[token] for token in sorted(content_expected)[:4]],
        )
    )
    similarity = SequenceMatcher(None, expected_tokens, actual_tokens).ratio()
    components.append(
        _component(
            "sentence_similarity",
            "структура всего предложения",
            0.15,
            similarity,
            "Косвенная речь должна сохранять участников и смысл, но менять грамматическую структуру.",
            f"Сравните свой вариант с «{expected}»: какая часть структуры отличается?",
            expected,
            [[expected]],
        )
    )

    is_request = "please" in _tokens(prompt) or (
        "asked" in expected_tokens and "to" in expected_tokens and "if" not in expected_tokens
    )
    if is_request:
        asked_index = expected_tokens.index("asked") if "asked" in expected_tokens else -1
        expected_object = (
            expected_tokens[asked_index + 1]
            if 0 <= asked_index < len(expected_tokens) - 1
            else "адресат"
        )
        to_index = expected_tokens.index("to") if "to" in expected_tokens else -1
        expected_verb = (
            expected_tokens[to_index + 1]
            if 0 <= to_index < len(expected_tokens) - 1
            else ""
        )
        request_score = 1.0 if "asked" in actual_tokens else (0.35 if "said" in actual_tokens else 0.0)
        components.append(
            _component(
                "request_reporting_verb",
                "передача просьбы через ask + object + to-infinitive",
                0.25,
                request_score,
                "Вежливая просьба в косвенной речи передаётся конструкцией ask + object + to-infinitive.",
                "Какой reporting verb и какую конструкцию нужно использовать, чтобы передать вежливую просьбу с please?",
                f"Нужно использовать ask + адресат + to-infinitive: {expected}.",
                [
                    ["ask", "asked", "попрос"],
                    ["object + to-infinitive", f"asked {expected_object} to", "адресат + to"],
                ],
            )
        )
        infinitive_score = 1.0 if expected_verb and _contains_tokens(
            actual, f"to {expected_verb}"
        ) else 0.0
        components.append(
            _component(
                "request_infinitive",
                "инфинитив после адресата просьбы",
                0.15,
                infinitive_score,
                "После ask и адресата используется to-infinitive.",
                "Какая форма глагола ставится после адресата в конструкции ask someone ...?",
                "После адресата ставится to-infinitive.",
                [["to-infinitive", "to + глагол", "инфинитив"]],
            )
        )
        prompt_the_pairs = {
            f"the {tokens[index + 1]}"
            for tokens in [_tokens(prompt)]
            for index, token in enumerate(tokens[:-1])
            if token == "the"
        }
        expected_the_pairs = {
            f"the {tokens[index + 1]}"
            for tokens in [expected_tokens]
            for index, token in enumerate(tokens[:-1])
            if token == "the"
        }
        unchanged_pairs = sorted(prompt_the_pairs & expected_the_pairs)
        if unchanged_pairs:
            target_pair = unchanged_pairs[0]
            determiner_score = 1.0 if _contains_tokens(actual, target_pair) else 0.0
            components.append(
                _component(
                    "unchanged_determiner",
                    f"сохранение {target_pair} у уже определённого объекта",
                    0.10,
                    determiner_score,
                    "The не заменяется на that автоматически: указательное слово меняется только если оно было в прямой речи.",
                    f"Почему в этой просьбе сохраняется {target_pair}, а не that + noun?",
                    f"В прямой речи уже было {target_pair}; правила reported speech не требуют заменять the на that.",
                    [[target_pair], ["the не меняется", "не заменяется на that"]],
                )
            )
        required_references = {
            token
            for token in expected_tokens
            if token in {"me", "him", "her", "us", "them"} or token == expected_object
        }
        if required_references:
            reference_score = sum(token in actual_tokens for token in required_references) / len(
                required_references
            )
            components.append(
                _component(
                    "pronouns",
                    "изменение местоимений и адресата по участникам ситуации",
                    0.10,
                    reference_score,
                    "Местоимения и адресат меняются с учётом говорящего и участников просьбы.",
                    "Как должны измениться говорящий, адресат и местоимения при переходе к косвенной речи?",
                    f"Ориентир: {expected}.",
                    [[token] for token in sorted(required_references)],
                )
            )
    else:
        reporting_score = 1.0 if any(token in actual_tokens for token in ["said", "asked", "told"]) else 0.0
        components.append(
            _component(
                "reporting_structure",
                "глагол сообщения и структура косвенной речи",
                0.10,
                reporting_score,
                "Косвенная речь вводится reporting verb и придаточной частью.",
                "Как в этом предложении вводится косвенная речь?",
                expected,
                [["said", "asked", "told", "сказал", "спросил"]],
            )
        )
        if "asked" in expected_tokens and any(token in expected_tokens for token in ["if", "whether"]):
            connector_score = 1.0 if any(token in actual_tokens for token in ["if", "whether"]) else 0.0
            components.append(
                _component(
                    "reported_question_connector",
                    "if/whether и прямой порядок слов",
                    0.15,
                    connector_score,
                    "Общий косвенный вопрос вводится if/whether и использует прямой порядок слов.",
                    "Чем вводится общий вопрос в косвенной речи и какой порядок слов используется?",
                    "Используется if или whether, затем прямой порядок слов.",
                    [["if", "whether", "ли"], ["прямой порядок", "statement order"]],
                )
            )

        backshift_patterns = [
            ("had", "Past Simple или Present Perfect после reporting verb в прошлом обычно сдвигается в Past Perfect."),
            ("was", "Present Continuous после reporting verb в прошлом обычно сдвигается в Past Continuous."),
            ("were", "Present после reporting verb в прошлом обычно сдвигается в Past."),
            ("would", "Will в косвенной речи после reporting verb в прошлом обычно меняется на would."),
            ("could", "Can в косвенной речи после reporting verb в прошлом обычно меняется на could."),
        ]
        required_aux = next((aux for aux, _ in backshift_patterns if aux in expected_tokens), "")
        if required_aux:
            explanation = next(text for aux, text in backshift_patterns if aux == required_aux)
            components.append(
                _component(
                    "backshift",
                    "сдвиг времени после reporting verb в прошлом",
                    0.25,
                    1.0 if required_aux in actual_tokens else 0.0,
                    explanation,
                    f"Как должна измениться глагольная форма в «{prompt}» после reporting verb в прошлом?",
                    f"Нужна форма с {required_aux}: {expected}.",
                    [[required_aux], ["сдвиг времени", "backshift", "прошедшее время"]],
                )
            )

        reference_mappings = [
            ("today", "that day", "today в косвенной речи с прошедшей точкой отсчёта обычно меняется на that day"),
            ("yesterday", "the day before", "yesterday обычно меняется на the day before"),
            ("this", "that", "this обычно меняется на that"),
            ("here", "there", "here обычно меняется на there"),
            ("tomorrow", "the next day", "tomorrow обычно меняется на the next day"),
        ]
        active_mapping = next(
            (
                (source, target, focus)
                for source, target, focus in reference_mappings
                if source in _tokens(prompt) and _contains_tokens(expected, target)
            ),
            None,
        )
        if active_mapping:
            source, target, focus = active_mapping
            reference_score = 1.0 if _contains_tokens(actual, target) else 0.0
            if target == "the day before" and _contains_tokens(actual, "the day before yesterday"):
                reference_score = 0.5
            components.append(
                _component(
                    "reference_shift",
                    f"замена {source} на {target}",
                    0.25,
                    reference_score,
                    focus + ".",
                    f"В прямой речи есть {source}, а reporting verb стоит в прошлом. На какую форму меняется {source} и почему?",
                    f"{source} меняется на {target}, потому что точка отсчёта переносится к моменту исходной речи.",
                    [[source], [target], ["точка отсчёта", "момент речи", "перенос времени"]],
                )
            )

    total_weight = sum(float(item["weight"]) for item in components)
    score = sum(float(item["weight"]) * float(item["score"]) for item in components) / total_weight
    errors = [item for item in components if float(item["score"]) < 0.99]
    primary_error = max(errors, key=lambda item: float(item["weight"]) * (1 - float(item["score"])), default=None)
    return {"score": round(score, 4), "components": components, "primary_error": primary_error}


SLOT_COMPONENT_GRADERS = {
    "eng_reported_speech": grade_reported_speech_slot,
}


def grade_numbered_answer(assignment: dict[str, Any], answer: str) -> dict[str, Any] | None:
    reference = str((assignment.get("rubric") or {}).get("reference_answer") or "")
    reference_items = parse_numbered_items(reference)
    if len(reference_items) < 2:
        return None
    answer_items = parse_numbered_items(answer)
    instruction_items = parse_numbered_items(str(assignment.get("instructions") or ""))
    skill_id = str((assignment.get("skill_ids") or [""])[0])
    slots = []
    for index, expected in enumerate(reference_items, start=1):
        actual = answer_items[index - 1] if index <= len(answer_items) else ""
        variants = accepted_variants(expected)
        correct = answer_matches(actual, variants)
        component_grader = SLOT_COMPONENT_GRADERS.get(skill_id)
        component_result = (
            component_grader(
                instruction_items[index - 1] if index <= len(instruction_items) else "",
                actual,
                variants[0],
            )
            if component_grader and not correct
            else None
        )
        slot_score = 1.0 if correct else float((component_result or {}).get("score", 0.0))
        primary_error = (component_result or {}).get("primary_error")
        if correct:
            issue = ""
        elif not actual:
            issue = f"Пункт {index} не заполнен."
        elif primary_error:
            issue = (
                f'Освоено {slot_score:.0%} пункта. Нужно исправить компонент: '
                f'{primary_error["label"]}. {primary_error["rule_focus"]}'
            )
        else:
            issue = f"В пункте {index} получено «{actual}», ожидается «{expected}»."
        slots.append(
            {
                "position": index,
                "prompt": instruction_items[index - 1] if index <= len(instruction_items) else "",
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "score": round(slot_score, 4),
                "components": (component_result or {}).get("components", []),
                "diagnostic": primary_error,
                "student_evidence": actual or "Ответ отсутствует",
                "expected_phrase": " / ".join(variants[:3]),
                "issue": issue,
            }
        )
    correct_count = sum(bool(slot["correct"]) for slot in slots)
    return {
        "source": (
            "rule_component_grader"
            if component_grader
            else "deterministic_answer_key"
        ),
        "correct": correct_count == len(slots),
        "score": sum(float(slot["score"]) for slot in slots) / len(slots),
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
