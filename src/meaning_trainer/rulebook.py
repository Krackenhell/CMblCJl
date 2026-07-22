from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = PROJECT_ROOT / "data" / "english_b2_rules.json"


def load_rulebook(path: Path = DEFAULT_RULES_PATH) -> dict[str, dict[str, Any]]:
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {str(rule["id"]): rule for rule in rules}


def rules_for_assignment(
    assignment: dict[str, Any], rulebook: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [rulebook[skill_id] for skill_id in assignment["skill_ids"] if skill_id in rulebook]
