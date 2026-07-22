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
GRAMMAR_PORT = int(os.getenv("VIVATRACE_GRAMMAR_PORT", "8082"))


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


def _relative_clause_findings(text: str, rule_id: str) -> list[dict[str, Any]]:
    if rule_id != "eng_relative_clauses":
        return []
    direct_verb_after_where = re.search(
        r"\bwhere\s+(is|are|was|were|has|have|had|does|did|attracts|serves|offers|"
        r"provides|contains|includes|causes|makes|seems|looks|becomes|works|lives|"
        r"stands|lies|remains|takes|gives|helps|allows|creates|means|shows|needs|uses)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not direct_verb_after_where:
        return []
    replacement = re.sub(r"\bwhere\b", "which", text, count=1, flags=re.IGNORECASE)
    return [
        {
            "source": "VivaTrace structural rule",
            "code": "RELATIVE_WHERE_MISSING_SUBJECT",
            "message": (
                "После relative adverb 'where' требуется отдельное подлежащее; "
                "если место само выполняет действие, нужен 'which'."
            ),
            "fragment": direct_verb_after_where.group(0),
            "suggestions": [replacement],
        }
    ]


def offline_grammar_findings(text: str, rule_id: str = "") -> list[dict[str, Any]]:
    findings = [*_language_tool_findings(text), *_relative_clause_findings(text, rule_id)]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in findings:
        unique[(str(finding.get("code")), str(finding.get("fragment")))] = finding
    return list(unique.values())
