from __future__ import annotations

import re
import unicodedata
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

RELATIVE_WORDS = {"who", "whom", "whose", "which", "that", "where"}
CONTENT_IRREGULARS = {
    "met": "meet",
    "won": "win",
    "was": "be",
    "were": "be",
    "is": "be",
    "are": "be",
    "has": "have",
    "had": "have",
}


def normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("’", "'").replace("`", "'")
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


def _content_lemma(token: str) -> str:
    token = CONTENT_IRREGULARS.get(token, token)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        return stem + "e" if stem + "e" in {"create"} else stem
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _content_lemmas(value: str) -> set[str]:
    stop_words = CONTENT_STOP_WORDS | RELATIVE_WORDS | {
        "is", "are", "be", "been", "being", "do", "does", "did",
        "still", "perfectly", "there", "only", "once", "many", "our", "my",
    }
    return {
        _content_lemma(token)
        for token in _tokens(value)
        if token not in stop_words and len(token) > 1
    }


def _relative_marker_and_antecedent(value: str) -> tuple[str, str]:
    tokens = _tokens(value)
    for index, token in enumerate(tokens):
        if token in {"which", "whom"} and index and tokens[index - 1] in {"in", "at", "for", "to"}:
            return f"{tokens[index - 1]} {token}", tokens[index - 2] if index >= 2 else ""
        if token in RELATIVE_WORDS:
            return token, tokens[index - 1] if index else ""
    return "", ""


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


def grade_relative_clause_slot(prompt: str, actual: str, expected: str) -> dict[str, Any]:
    """Score a relative-clause answer by meaning and grammar, not one exact sentence."""
    if not _tokens(actual):
        return {
            "score": 0.0,
            "components": [],
            "primary_error": None,
            "probe": None,
            "accepted": False,
        }

    expected_marker, expected_antecedent = _relative_marker_and_antecedent(expected)
    actual_marker, actual_antecedent = _relative_marker_and_antecedent(actual)
    expected_marker_core = expected_marker.split()[-1]
    actual_marker_core = actual_marker.split()[-1] if actual_marker else ""
    non_defining_marker = r",\s*(?:who|whom|whose|which|where)\b"
    expected_non_defining = bool(
        re.search(non_defining_marker, expected, re.IGNORECASE)
    )
    actual_non_defining = bool(
        re.search(non_defining_marker, actual, re.IGNORECASE)
    )

    expected_content = _content_lemmas(expected)
    actual_content = _content_lemmas(actual)
    content_score = (
        len(expected_content & actual_content) / len(expected_content)
        if expected_content
        else 1.0
    )
    content_component = _component(
        "meaning_preservation",
        "сохранение фактов из двух исходных предложений",
        0.25,
        content_score,
        "При объединении предложений нельзя терять факт или менять того, кто совершает действие.",
        f"Какие два исходных факта из «{prompt}» должны сохраниться в одном предложении?",
        f"Нужно сохранить оба факта без изменения смысла: {expected}.",
        [[token] for token in sorted(expected_content)[:5]],
    )

    allowed_markers: set[str]
    if expected_marker_core in {"who", "whom"}:
        allowed_markers = {expected_marker_core} if expected_non_defining else {"who", "whom", "that"}
        marker_rule = "who/whom относится к людям; that возможно только в defining clause"
        marker_concepts = [
            ["who", "whom", "that"],
            ["человек", "люди", "person", "people"],
            ["определительное придаточное", "relative clause"],
            ["потому что", "так как", "refers to", "относится к"],
        ]
    elif expected_marker_core == "whose":
        allowed_markers = {"whose"}
        marker_rule = "whose показывает принадлежность"
        marker_concepts = [
            ["whose"],
            ["принадлежность", "possessive", "чей"],
            ["определительное придаточное", "relative clause"],
        ]
    elif expected_marker_core == "where":
        allowed_markers = {"where", "which", "that"}
        marker_rule = "where заменяет обстоятельство места; также возможна равнозначная перестройка с in which/that"
        marker_concepts = [
            ["where", "in which", "that", "which"],
            ["место", "place", "там", "there"],
            ["определительное придаточное", "relative clause"],
            ["потому что", "так как", "refers to", "относится к"],
        ]
    else:
        allowed_markers = {"which"} if expected_non_defining else {"which", "that"}
        marker_rule = "which относится к предметам; that возможно только в defining clause"
        marker_concepts = [
            ["which", "that"],
            ["предмет", "вещь", "thing", "object"],
            ["определительное придаточное", "relative clause"],
            ["потому что", "так как", "refers to", "относится к"],
        ]

    place_rewrite = False
    if expected_marker_core == "where" and actual_marker_core in {"which", "that"}:
        antecedent_pattern = re.escape(actual_antecedent)
        place_rewrite = bool(
            re.search(
                rf"\b(?:in|at)\s+(?:the\s+|a\s+|this\s+|that\s+)?{antecedent_pattern}\b",
                normalize_answer(actual),
            )
        ) or actual_marker.startswith("in ")
    marker_valid = actual_marker_core in allowed_markers
    if expected_marker_core == "where" and actual_marker_core in {"which", "that"}:
        marker_valid = marker_valid and place_rewrite
    marker_score = 1.0 if marker_valid else (0.5 if actual_marker_core in RELATIVE_WORDS else 0.0)
    marker_component = _component(
        "relative_marker",
        "выбор относительного слова по роли существительного",
        0.15,
        marker_score,
        marker_rule + ".",
        f"Какую роль выполняет «{expected_antecedent}» и какое относительное слово передаёт эту роль?",
        f"Ориентир: {marker_rule}. В данном пункте: {expected}.",
        marker_concepts,
    )

    antecedent_score = 1.0 if (
        _content_lemma(actual_antecedent) == _content_lemma(expected_antecedent)
    ) else 0.0
    antecedent_component = _component(
        "antecedent",
        "правильная связь придаточного с определяемым словом",
        0.25,
        antecedent_score,
        "Относительное придаточное относится к существительному прямо перед связкой; неверная связь меняет участника действия.",
        f"В вашем варианте относительная часть присоединена к «{actual_antecedent or '—'}». К какому слову она должна относиться, чтобы сохранить смысл «{prompt}»?",
        f"Она должна относиться к «{expected_antecedent}»: {expected}.",
        [
            [expected_antecedent],
            ["относится к", "определяет", "refers to", "describes"],
            ["смысл", "meaning", "кто выполняет действие"],
        ],
    )

    if expected_non_defining:
        clause_type_score = 1.0 if actual_non_defining and actual_marker_core != "that" else 0.0
        clause_rule = (
            "Уже определённое существительное получает дополнительную non-defining информацию; "
            "она выделяется запятыми, и that здесь не используется"
        )
        clause_concepts = [
            ["дополнительная информация", "необязательная информация", "non-defining"],
            ["запятая", "запятые", "comma", "commas"],
            ["which", "who"],
            [
                "that не используется",
                "нельзя that",
                "that здесь нельзя",
                "that is not allowed",
                "not that",
            ],
        ]
    else:
        clause_type_score = 1.0 if not actual_non_defining else 0.6
        clause_rule = (
            "Defining relative clause уточняет, о каком человеке, предмете или месте "
            "идёт речь, и пишется без запятых"
        )
        clause_concepts = [
            ["уточняет", "определяет", "defining"],
            ["без запятых", "no commas"],
            [expected_marker_core],
        ]
    clause_component = _component(
        "clause_type",
        "defining/non-defining смысл и пунктуация",
        0.25,
        clause_type_score,
        clause_rule + ".",
        f"Является ли относительная часть в «{expected}» необходимым уточнением или дополнительной информацией, и нужны ли запятые?",
        clause_rule + f" Правильная форма: {expected}.",
        clause_concepts,
    )

    structure_score = 1.0 if actual_marker and len(_tokens(actual)) >= 5 else 0.0
    structure_component = _component(
        "complete_clause",
        "полное предложение с относительным придаточным",
        0.10,
        structure_score,
        "Ответ должен быть одним полным предложением с относительным придаточным.",
        "Как объединить обе исходные части в одно полное предложение с relative clause?",
        expected,
        [[expected]],
    )
    components = [
        content_component,
        marker_component,
        antecedent_component,
        clause_component,
        structure_component,
    ]
    score = sum(float(item["weight"]) * float(item["score"]) for item in components)
    errors = [item for item in components if float(item["score"]) < 0.99]
    primary_error = max(
        errors,
        key=lambda item: float(item["weight"]) * (1 - float(item["score"])),
        default=None,
    )
    minimal_correction = expected
    if (
        primary_error
        and primary_error.get("code") == "clause_type"
        and not expected_non_defining
        and actual_non_defining
    ):
        # When the only verified problem is defining/non-defining punctuation,
        # keep the student's vocabulary and information order. Replacing a/an
        # or reordering the whole sentence would falsely suggest extra errors.
        minimal_correction = re.sub(r"\s*,\s*", " ", actual).strip()
        primary_error = dict(primary_error)
        primary_error["expected_answer"] = (
            clause_rule + f" Исправьте только структуру: {minimal_correction}."
        )
    probe = clause_component if expected_non_defining else marker_component
    accepted = all(
        (
            content_score >= 0.90,
            marker_valid,
            antecedent_score >= 0.99,
            clause_type_score >= 0.99,
            structure_score >= 0.99,
        )
    )
    return {
        "score": round(score, 4),
        "components": components,
        "primary_error": primary_error,
        "probe": probe,
        "accepted": accepted,
        "minimal_correction": minimal_correction,
    }


SLOT_COMPONENT_GRADERS = {
    "eng_reported_speech": grade_reported_speech_slot,
    "eng_relative_clauses": grade_relative_clause_slot,
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
        exact_match = answer_matches(actual, variants)
        component_grader = SLOT_COMPONENT_GRADERS.get(skill_id)
        component_result = (
            component_grader(
                instruction_items[index - 1] if index <= len(instruction_items) else "",
                actual,
                variants[0],
            )
            if component_grader
            else None
        )
        accepted_equivalent = bool((component_result or {}).get("accepted"))
        correct = exact_match or accepted_equivalent
        slot_score = 1.0 if correct else float((component_result or {}).get("score", 0.0))
        primary_error = (component_result or {}).get("primary_error")
        probe = (component_result or {}).get("probe")
        if not probe:
            probe = next(
                (
                    item
                    for item in (component_result or {}).get("components", [])
                    if item.get("code") not in {"content", "sentence_similarity"}
                ),
                None,
            )
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
                "probe": probe,
                "accepted_equivalent": accepted_equivalent and not exact_match,
                "student_evidence": actual or "Ответ отсутствует",
                "expected_phrase": " / ".join(variants[:3]),
                "correction": str(
                    (component_result or {}).get("minimal_correction")
                    or " / ".join(variants[:3])
                ),
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
