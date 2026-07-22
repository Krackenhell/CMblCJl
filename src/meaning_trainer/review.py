from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def review_interval_days(score: float) -> int:
    score = min(max(float(score), 0.0), 1.0)
    if score < 0.5:
        return 1
    if score < 0.75:
        return 3
    if score < 0.9:
        return 7
    return 14


def build_review_plan(
    records: list[dict[str, Any]], now: datetime | None = None
) -> list[dict[str, Any]]:
    """Build a transparent spacing plan from latest scored learning events."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    latest_by_topic: dict[str, dict[str, Any]] = {}
    for record in records:
        topic_key = str(record.get("topic_key") or record.get("skill_id") or "")
        completed_at = str(record.get("completed_at") or record.get("updated_at") or "")
        if not topic_key or not completed_at:
            continue
        previous = latest_by_topic.get(topic_key)
        if previous is None or completed_at > str(
            previous.get("completed_at") or previous.get("updated_at") or ""
        ):
            latest_by_topic[topic_key] = record

    plan: list[dict[str, Any]] = []
    for topic_key, record in latest_by_topic.items():
        timestamp = str(record.get("completed_at") or record.get("updated_at"))
        completed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=UTC)
        score = float(
            record.get("score")
            if record.get("score") is not None
            else record.get("overall_score")
            if record.get("overall_score") is not None
            else record.get("submission_score")
            or 0
        )
        interval = review_interval_days(score)
        due = completed + timedelta(days=interval)
        days_left = (due.date() - now.date()).days
        status = "сегодня" if days_left == 0 else "просрочено" if days_left < 0 else "запланировано"
        plan.append(
            {
                "topic_key": topic_key,
                "title": str(record.get("topic") or record.get("title") or topic_key),
                "score": round(score, 4),
                "due_at": due.isoformat(),
                "days_left": days_left,
                "status": status,
                "interval_days": interval,
            }
        )
    return sorted(plan, key=lambda item: (item["days_left"], item["score"]))
