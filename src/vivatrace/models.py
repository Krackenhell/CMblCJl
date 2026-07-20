from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Route(StrEnum):
    REPAIR = "repair"
    PRACTICE = "practice"
    TRANSFER = "transfer"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    prerequisites: tuple[str, ...] = ()
    target_mastery: float = 0.80


@dataclass(frozen=True)
class Curriculum:
    course_id: str
    course_name: str
    topic_id: str
    topic_name: str
    learning_goal: str
    skills: tuple[Skill, ...]

    @property
    def skill_by_id(self) -> dict[str, Skill]:
        return {skill.id: skill for skill in self.skills}


@dataclass(frozen=True)
class ProbeQuestion:
    id: str
    skill_id: str
    text: str
    purpose: str
    expected_concepts: tuple[tuple[str, ...], ...]
    misconception_patterns: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    skill_id: str
    score: float
    confidence: float
    quote: str
    rationale: str
    misconception: str | None = None
    source: str = "viva"
    evaluator_model: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class ArtifactFinding:
    skill_id: str
    severity: str
    evidence: str
    hypothesis: str
    confidence: float


@dataclass
class StudentState:
    student_id: str
    name: str
    mastery: dict[str, float]
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recommendation:
    route: Route
    title: str
    action: str
    duration_minutes: int
    skill_id: str
