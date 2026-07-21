from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MISSIONS_PATH = PROJECT_ROOT / "data" / "english_b2_missions.json"


FEATURE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "indefinite_article": (
        re.compile(r"\b(?:a|an)\s+[a-z][a-z'-]*\b", re.IGNORECASE),
    ),
    "definite_article": (
        re.compile(r"\bthe\s+[a-z][a-z'-]*\b", re.IGNORECASE),
    ),
    "present_perfect": (
        re.compile(
            r"\b(?:have|has)\s+(?:(?:never|ever|already|just)\s+)?"
            r"(?:been|done|seen|lost|visited|tried|finished|written|worked|made|taken|gone|sent|"
            r"learned|learnt|built|completed|created|developed)\b",
            re.IGNORECASE,
        ),
    ),
    "finished_past_time": (
        re.compile(r"\b(?:yesterday|last\s+(?:week|month|year)|\d+\s+(?:days?|years?)\s+ago|in\s+20\d{2})\b", re.IGNORECASE),
    ),
    "strong_deduction": (
        re.compile(r"\bmust\s+(?:be|have|know|mean|belong|need)\b", re.IGNORECASE),
    ),
    "uncertain_deduction": (
        re.compile(r"\b(?:might|could)\s+(?:be|have|know|mean|belong|need)\b", re.IGNORECASE),
    ),
    "stop_gerund": (
        re.compile(r"\bstop(?:ped|ping)?\s+[a-z][a-z'-]*ing\b", re.IGNORECASE),
    ),
    "plan_infinitive": (
        re.compile(r"\b(?:plan|decide|agree|intend)(?:ned|d|s)?\s+to\s+[a-z][a-z'-]*\b", re.IGNORECASE),
    ),
    "first_conditional": (
        re.compile(r"\bif\b[^.!?]{1,120}\bwill\b", re.IGNORECASE),
    ),
    "second_conditional": (
        re.compile(r"\bif\b[^.!?]{1,120}\bwould\b", re.IGNORECASE),
    ),
    "past_passive": (
        re.compile(
            r"\b(?:was|were|has\s+been|have\s+been|had\s+been)\s+"
            r"(?:[a-z][a-z'-]*ed|built|done|made|known|sent|written|stolen|shown)\b",
            re.IGNORECASE,
        ),
    ),
    "future_passive": (
        re.compile(
            r"\bwill\s+be\s+(?:[a-z][a-z'-]*ed|built|done|made|known|sent|written|shown)\b",
            re.IGNORECASE,
        ),
    ),
    "reported_request": (
        re.compile(r"\basked\s+(?:me|you|him|her|us|them|[A-Z][a-z]+)\s+to\s+[a-z]+\b", re.IGNORECASE),
    ),
    "reported_time_shift": (
        re.compile(r"\b(?:that\s+day|the\s+day\s+before|the\s+next\s+day)\b", re.IGNORECASE),
    ),
    "relative_person": (re.compile(r"\bwho\b", re.IGNORECASE),),
    "relative_place": (re.compile(r"\bwhere\b", re.IGNORECASE),),
    "contrast_linker": (
        re.compile(r"\b(?:however|although|whereas|nevertheless|on\s+the\s+other\s+hand)\b", re.IGNORECASE),
    ),
    "result_linker": (
        re.compile(r"\b(?:therefore|thus|consequently|as\s+a\s+result|in\s+conclusion|overall)\b", re.IGNORECASE),
    ),
    "formal_request": (
        re.compile(
            r"\b(?:would\s+it\s+be\s+possible|could\s+you\s+please|i\s+would\s+appreciate|"
            r"i\s+am\s+writing\s+to|may\s+i\s+ask)\b",
            re.IGNORECASE,
        ),
    ),
    "formal_reason": (
        re.compile(r"\b(?:due\s+to|owing\s+to|because\s+of|as\s+i\s+am|unfortunately)\b", re.IGNORECASE),
    ),
}


def load_missions(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or DEFAULT_MISSIONS_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Банк практических миссий пуст или имеет неверный формат.")
    required = {
        "id", "topic_key", "skill_id", "title", "setting", "npc_name", "npc_role",
        "opening", "objective", "student_brief", "required_features", "max_turns",
        "success_threshold",
    }
    identifiers: set[str] = set()
    topics: set[str] = set()
    for mission in payload:
        missing = required - set(mission)
        if missing:
            raise ValueError(f'Миссия {mission.get("id", "—")}: отсутствуют поля {sorted(missing)}')
        if mission["id"] in identifiers:
            raise ValueError(f'Повторяющийся id миссии: {mission["id"]}')
        identifiers.add(str(mission["id"]))
        topics.add(str(mission["topic_key"]))
        feature_ids = [str(item.get("id")) for item in mission["required_features"]]
        unknown = [feature_id for feature_id in feature_ids if feature_id not in FEATURE_PATTERNS]
        if unknown:
            raise ValueError(f'Миссия {mission["id"]}: неизвестные признаки {unknown}')
        if not 1 <= int(mission["max_turns"]) <= 5:
            raise ValueError(f'Миссия {mission["id"]}: max_turns должен быть от 1 до 5')
        if not 0.5 <= float(mission["success_threshold"]) <= 1:
            raise ValueError(f'Миссия {mission["id"]}: неверный success_threshold')
    if len(topics) != len(payload):
        raise ValueError("В MVP должна быть одна однозначная миссия на каждую тему.")
    return payload


def missions_by_topic(missions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(mission["topic_key"]): mission for mission in missions}


def detect_mission_features(
    mission: dict[str, Any], student_messages: list[str]
) -> dict[str, Any]:
    text = "\n".join(message.strip() for message in student_messages if message.strip())
    features: list[dict[str, Any]] = []
    for required in mission["required_features"]:
        feature_id = str(required["id"])
        evidence: list[str] = []
        for pattern in FEATURE_PATTERNS[feature_id]:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if value and value.lower() not in {item.lower() for item in evidence}:
                    evidence.append(value)
        features.append(
            {
                "id": feature_id,
                "label": str(required["label"]),
                "found": bool(evidence),
                "evidence": evidence[:3],
            }
        )
    found = sum(bool(item["found"]) for item in features)
    coverage = found / len(features) if features else 0.0
    return {
        "coverage": round(coverage, 4),
        "features": features,
        "found": [item["label"] for item in features if item["found"]],
        "missing": [item["label"] for item in features if not item["found"]],
    }


def _normalized_fragment(value: str) -> str:
    value = value.casefold().replace("’", "'").replace("`", "'")
    value = re.sub(r"[^a-zа-яё0-9']+", " ", value)
    return " ".join(value.split())


def _english_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", value.casefold())


def _is_local_correction(fragment: str, correction: str) -> bool:
    source = set(_english_tokens(fragment))
    target = set(_english_tokens(correction))
    if not source or not target:
        return False
    overlap = len(source & target) / min(len(source), len(target))
    return overlap >= 0.3


def _invalid_first_mention_article_swap(
    fragment: str, correction: str, student_message: str
) -> bool:
    pattern = r"(?=\b{article}\s+(?:[a-z][a-z'-]*\s+){{0,2}}([a-z][a-z'-]*)\b)"
    source_nouns = {
        item.casefold()
        for item in re.findall(pattern.format(article="(?:a|an)"), fragment, re.IGNORECASE)
    }
    target_nouns = {
        item.casefold()
        for item in re.findall(pattern.format(article="the"), correction, re.IGNORECASE)
    }
    swapped_nouns = source_nouns & target_nouns
    if not swapped_nouns:
        return False
    fragment_position = student_message.casefold().find(fragment.casefold())
    prefix = student_message[: max(fragment_position, 0)]
    return any(
        not re.search(rf"\b{re.escape(noun)}\b", prefix, re.IGNORECASE)
        for noun in swapped_nouns
    )


def _mission_specific_false_positive(error: dict[str, Any], mission: dict[str, Any] | None) -> bool:
    if not mission:
        return False
    fragment = str(error.get("fragment") or "")
    correction = str(error.get("correction") or "")
    feature_ids = {str(item["id"]) for item in mission.get("required_features") or []}
    if {"stop_gerund", "plan_infinitive"} <= feature_ids:
        valid_stop = FEATURE_PATTERNS["stop_gerund"][0].search(fragment)
        valid_plan = FEATURE_PATTERNS["plan_infinitive"][0].search(fragment)
        if valid_stop and valid_plan and (
            "plan to decide to" in correction.casefold()
            or "stop doing" in str(error.get("explanation") or "").casefold()
        ):
            return True
    if {"relative_person", "relative_place"} <= feature_ids:
        valid_place_clause = re.search(
            r"\b(?:place|library|cafe|café|school|office|city|room)\s+where\s+"
            r"(?:i|you|he|she|we|they|people|students)\s+"
            r"(?:can|could|may|might|will|would|study|work|live|meet|go|stay|read)\b",
            fragment,
            re.IGNORECASE,
        )
        if valid_place_clause and "where" in correction.casefold():
            return True
    return False


def grounded_error_items(
    errors: list[dict[str, Any]],
    student_message: str,
    mission: dict[str, Any] | None = None,
) -> tuple[list[dict], int]:
    normalized_answer = f" {_normalized_fragment(student_message)} "
    verified: list[dict] = []
    discarded = 0
    for error in errors:
        fragment = str(error.get("fragment") or "").strip()
        normalized_fragment = _normalized_fragment(fragment)
        if not normalized_fragment or f" {normalized_fragment} " not in normalized_answer:
            discarded += 1
            continue
        explanation = str(error.get("explanation") or "").strip()
        correction = str(error.get("correction") or "").strip()
        normalized_explanation = _normalized_fragment(explanation)
        normalized_correction = _normalized_fragment(correction)
        if (
            not explanation
            or not correction
            or normalized_explanation == normalized_fragment
            or normalized_correction == normalized_fragment
            or normalized_explanation == normalized_correction
            or not _is_local_correction(fragment, correction)
            or _invalid_first_mention_article_swap(fragment, correction, student_message)
            or _mission_specific_false_positive(error, mission)
        ):
            discarded += 1
            continue
        verified.append(
            {
                "fragment": fragment,
                "explanation": explanation,
                "correction": correction,
            }
        )
    return verified, discarded


def validate_mission_evaluation(
    mission: dict[str, Any],
    raw: dict[str, Any],
    student_message: str,
    signal: dict[str, Any],
    turn_count: int,
) -> dict[str, Any]:
    errors, discarded = grounded_error_items(
        list(raw.get("errors") or []), student_message, mission
    )
    grammar_score = min(max(float(raw.get("grammar_score") or 0), 0.0), 1.0)
    communicative_score = min(max(float(raw.get("communicative_score") or 0), 0.0), 1.0)
    coverage = float(signal["coverage"])
    if coverage <= 0:
        grammar_score = min(grammar_score, 0.35)
    elif coverage < 1:
        grammar_score = min(grammar_score, 0.70)
    score = 0.45 * coverage + 0.30 * grammar_score + 0.25 * communicative_score
    if coverage <= 0:
        score = min(score, 0.45)
    elif coverage < 1:
        score = min(score, 0.74)
    score = round(min(max(score, 0.0), 1.0), 4)
    success = (
        coverage >= 0.99
        and score >= float(mission["success_threshold"])
        and (
            bool(raw.get("ready_to_finish"))
            or communicative_score >= 0.85
        )
        and not errors
    )
    exhausted = turn_count >= int(mission["max_turns"]) and not success
    status = "completed" if success else "needs_retry" if exhausted else "active"
    return {
        "npc_reply": str(raw.get("npc_reply") or "").strip(),
        "positive_feedback": str(raw.get("positive_feedback") or "").strip(),
        "guidance": str(raw.get("guidance") or "").strip(),
        "errors": errors,
        "discarded_error_count": discarded,
        "requires_review": discarded > 0,
        "grammar_score": round(grammar_score, 4),
        "communicative_score": round(communicative_score, 4),
        "score": score,
        "state_summary": str(raw.get("state_summary") or "").strip(),
        "suggested_next_action": str(raw.get("suggested_next_action") or "").strip(),
        "signal": signal,
        "success": success,
        "status": status,
    }
