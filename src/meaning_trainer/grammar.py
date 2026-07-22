from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_PORT = int(os.getenv("MEANING_GRAMMAR_PORT", "8082"))


def _first_path(pattern: str) -> Path | None:
    return next(PROJECT_ROOT.glob(pattern), None)


def grammar_runtime_identity() -> dict[str, Any]:
    java = _first_path("tools/jre/**/java.exe")
    server = _first_path("tools/languagetool/**/languagetool-server.jar")
    missing = []
    if java is None:
        missing.append("tools/jre/**/java.exe")
    if server is None:
        missing.append("tools/languagetool/**/languagetool-server.jar")
    return {
        "ready": not missing,
        "missing": missing,
        "java": java,
        "server": server,
        "name": "LanguageTool 6.6 offline",
    }


def grammar_server_ready(port: int = GRAMMAR_PORT) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/v2/languages", timeout=0.6
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_grammar_server(port: int = GRAMMAR_PORT) -> dict[str, Any]:
    identity = grammar_runtime_identity()
    if not identity["ready"]:
        return {**identity, "server_ready": False, "port": port}
    if grammar_server_ready(port):
        return {**identity, "server_ready": True, "port": port}

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout = (log_dir / "grammar-server.stdout.log").open("a", encoding="utf-8")
    stderr = (log_dir / "grammar-server.stderr.log").open("a", encoding="utf-8")
    java = Path(identity["java"]).resolve()
    server = Path(identity["server"]).resolve()
    subprocess.Popen(  # noqa: S603
        [
            str(java),
            "-Xms128m",
            "-Xmx512m",
            "-jar",
            str(server),
            "--port",
            str(port),
        ],
        cwd=server.parent,
        stdout=stdout,
        stderr=stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if grammar_server_ready(port):
            return {**identity, "server_ready": True, "port": port}
        time.sleep(0.25)
    return {**identity, "server_ready": False, "port": port}


def _language_tool_findings(text: str, port: int = GRAMMAR_PORT) -> list[dict[str, Any]]:
    if not grammar_server_ready(port) and not ensure_grammar_server(port)["server_ready"]:
        return []
    body = urllib.parse.urlencode({"language": "en-US", "text": text}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/v2/check",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []

    findings = []
    for match in payload.get("matches") or []:
        rule = match.get("rule") or {}
        if str(rule.get("issueType") or "") != "grammar":
            continue
        replacements = [
            str(item.get("value") or "").strip()
            for item in match.get("replacements") or []
            if item.get("value")
        ][:3]
        offset = int(match.get("offset") or 0)
        length = int(match.get("length") or 0)
        suggestions = [
            text[:offset] + replacement + text[offset + length :]
            for replacement in replacements
        ]
        findings.append(
            {
                "source": "LanguageTool 6.6 offline",
                "code": str(rule.get("id") or "GRAMMAR"),
                "message": str(match.get("message") or "Grammar rule violation."),
                "fragment": text[offset : offset + length] or str(
                    (match.get("context") or {}).get("text") or text
                ),
                "suggestions": suggestions,
            }
        )
    return findings


_NON_HUMAN_RELATIVE_NOUNS = (
    "app|book|building|cafe|café|car|city|company|computer|course|country|"
    "food|hotel|idea|laptop|place|project|restaurant|room|school|sentence|"
    "phone|town|university|village|website"
)
_PERSON_RELATIVE_NOUNS = (
    "boy|designer|doctor|employee|friend|girl|manager|man|people|person|"
    "student|teacher|woman|worker"
)
_PLACE_RELATIVE_NOUNS = (
    "building|cafe|café|city|country|hotel|place|restaurant|room|school|"
    "town|university|village"
)


def relative_clause_evidence(text: str, rule_id: str = "") -> list[dict[str, Any]]:
    """Extract auditable relative-clause signals without asking an LLM to infer them."""
    if rule_id != "eng_relative_clauses":
        return []
    pattern = re.compile(
        r"\b(?P<antecedent>(?:(?:my|the|a|an|this|that|our|their|his|her)\s+)?"
        r"[A-Za-zÀ-ÿ'-]+)\s*(?P<comma>,?)\s+(?P<marker>who|whom|which|that|where)\b"
        r"(?P<tail>[^.!?]{0,100})",
        flags=re.IGNORECASE,
    )
    evidence: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        antecedent = match.group("antecedent").strip()
        noun = antecedent.lower().split()[-1]
        marker = match.group("marker").lower()
        comma = bool(match.group("comma"))
        if re.fullmatch(_PERSON_RELATIVE_NOUNS, noun, flags=re.IGNORECASE):
            category = "person"
        elif re.fullmatch(_PLACE_RELATIVE_NOUNS, noun, flags=re.IGNORECASE):
            category = "place"
        elif re.fullmatch(_NON_HUMAN_RELATIVE_NOUNS, noun, flags=re.IGNORECASE):
            category = "thing"
        else:
            category = "unknown"
        tail_words = re.findall(r"[A-Za-z']+", match.group("tail"))
        where_has_subject = marker != "where" or bool(
            tail_words
            and tail_words[0].lower()
            in {"i", "you", "he", "she", "it", "we", "they", "people", "students"}
        )
        marker_valid = not (
            marker in {"who", "whom"} and category in {"thing", "place"}
            or marker == "which" and category == "person"
            or marker == "where" and (category not in {"place", "unknown"} or not where_has_subject)
            or marker == "that" and comma
        )
        evidence.append(
            {
                "antecedent": antecedent,
                "category": category,
                "marker": marker,
                "clause_type_from_transcript": "non_defining" if comma else "defining",
                "marker_valid": marker_valid,
                "where_has_subject": where_has_subject,
                "punctuation_is_asr_evidence_only": True,
                "quote": match.group(0).strip(),
            }
        )
    return evidence


def _relative_clause_findings(text: str, rule_id: str) -> list[dict[str, Any]]:
    if rule_id != "eng_relative_clauses":
        return []
    findings: list[dict[str, Any]] = []
    direct_verb_after_where = re.search(
        r"\bwhere\s+(is|are|was|were|has|have|had|does|did|attracts|serves|offers|"
        r"provides|contains|includes|causes|makes|seems|looks|becomes|works|lives|"
        r"stands|lies|remains|takes|gives|helps|allows|creates|means|shows|needs|uses)\b",
        text,
        flags=re.IGNORECASE,
    )
    if direct_verb_after_where:
        replacement = re.sub(r"\bwhere\b", "which", text, count=1, flags=re.IGNORECASE)
        findings.append(
            {
                "source": "Meaning structural rule",
                "code": "RELATIVE_WHERE_MISSING_SUBJECT",
                "message": (
                    "После relative adverb 'where' требуется отдельное подлежащее; "
                    "если место само выполняет действие, нужен 'which'."
                ),
                "fragment": direct_verb_after_where.group(0),
                "minimal_correction": re.sub(
                    r"\bwhere\b",
                    "which",
                    direct_verb_after_where.group(0),
                    flags=re.IGNORECASE,
                ),
                "suggestions": [replacement],
            }
        )
    non_human_who = re.search(
        rf"\b(?P<noun>{_NON_HUMAN_RELATIVE_NOUNS})\s*,?\s+(?P<marker>who|whom)\b",
        text,
        flags=re.IGNORECASE,
    )
    if non_human_who:
        marker_start, marker_end = non_human_who.span("marker")
        replacement = text[:marker_start] + "which" + text[marker_end:]
        findings.append(
            {
                "source": "Meaning structural rule",
                "code": "RELATIVE_WHO_NON_HUMAN",
                "message": (
                    "Who/whom относится к людям. Если предмет или место само выполняет "
                    "действие, нужен which; where используется в значении in/at which."
                ),
                "fragment": non_human_who.group(0),
                "minimal_correction": "which",
                "suggestions": [replacement],
            }
        )
    person_which = re.search(
        rf"\b(?P<noun>{_PERSON_RELATIVE_NOUNS})\s*,?\s+(?P<marker>which)\b",
        text,
        flags=re.IGNORECASE,
    )
    if person_which:
        marker_start, marker_end = person_which.span("marker")
        replacement = text[:marker_start] + "who" + text[marker_end:]
        findings.append(
            {
                "source": "Meaning structural rule",
                "code": "RELATIVE_WHICH_PERSON",
                "message": "Для человека нужен who/whom, а не which.",
                "fragment": person_which.group(0),
                "minimal_correction": "who",
                "suggestions": [replacement],
            }
        )
    comma_that = re.search(r",\s+that\b", text, flags=re.IGNORECASE)
    if comma_that:
        marker_start = comma_that.start() + comma_that.group(0).lower().rfind("that")
        replacement = text[:marker_start] + "which" + text[marker_start + 4 :]
        findings.append(
            {
                "source": "Meaning structural rule",
                "code": "RELATIVE_THAT_NON_DEFINING",
                "message": "That не используется в non-defining clause после запятой.",
                "fragment": comma_that.group(0).strip(),
                "minimal_correction": "which",
                "suggestions": [replacement],
            }
        )
    return findings


def _spoken_usage_findings(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    works_good = re.search(r"\bworks?\s+good\b", text, flags=re.IGNORECASE)
    if works_good:
        replacement = text[: works_good.start()] + re.sub(
            r"good\b", "well", works_good.group(0), flags=re.IGNORECASE
        ) + text[works_good.end() :]
        findings.append(
            {
                "source": "Meaning usage rule",
                "code": "ADVERB_WORKS_WELL",
                "message": "После works для описания способа действия нужно наречие well.",
                "fragment": works_good.group(0),
                "minimal_correction": "works well",
                "suggestions": [replacement],
            }
        )
    represent_to_university = re.search(
        r"\brepresent\s+to\s+(?:[A-Za-z,. ]{0,40})?university\b",
        text,
        flags=re.IGNORECASE,
    )
    if represent_to_university:
        replacement = re.sub(r"\brepresent\b", "present", text, count=1, flags=re.IGNORECASE)
        findings.append(
            {
                "source": "Meaning usage rule",
                "code": "PRESENT_PROJECT_TO_UNIVERSITY",
                "message": (
                    "Для демонстрации проекта университету обычно используется present, "
                    "а не represent."
                ),
                "fragment": represent_to_university.group(0),
                "minimal_correction": re.sub(
                    r"\brepresent\b",
                    "present",
                    represent_to_university.group(0),
                    count=1,
                    flags=re.IGNORECASE,
                ),
                "suggestions": [replacement],
            }
        )
    return findings


def offline_grammar_findings(text: str, rule_id: str = "") -> list[dict[str, Any]]:
    findings = [
        *_language_tool_findings(text),
        *_relative_clause_findings(text, rule_id),
        *_spoken_usage_findings(text),
    ]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in findings:
        unique[(str(finding.get("code")), str(finding.get("fragment")))] = finding
    return list(unique.values())
