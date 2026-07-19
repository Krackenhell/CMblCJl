from __future__ import annotations

import json
from pathlib import Path

from .models import Curriculum, Skill


def load_curriculum(path: str | Path) -> Curriculum:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    skills = tuple(
        Skill(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            prerequisites=tuple(item.get("prerequisites", [])),
            target_mastery=float(item.get("target_mastery", 0.8)),
        )
        for item in payload["skills"]
    )
    return Curriculum(
        course_id=payload["course_id"],
        course_name=payload["course_name"],
        topic_id=payload["topic_id"],
        topic_name=payload["topic_name"],
        learning_goal=payload["learning_goal"],
        skills=skills,
    )
