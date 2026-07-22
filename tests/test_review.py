from __future__ import annotations

from datetime import UTC, datetime

from meaning_trainer.review import build_review_plan, review_interval_days


def test_review_intervals_are_shorter_for_weaker_evidence():
    assert [review_interval_days(score) for score in (0.2, 0.6, 0.8, 0.95)] == [1, 3, 7, 14]


def test_review_plan_keeps_latest_event_per_topic_and_marks_due_items():
    records = [
        {
            "topic_key": "eng_articles",
            "topic": "Артикли",
            "score": 0.4,
            "completed_at": "2026-07-01T10:00:00+00:00",
        },
        {
            "topic_key": "eng_articles",
            "topic": "Артикли",
            "score": 0.8,
            "completed_at": "2026-07-20T10:00:00+00:00",
        },
        {
            "topic_key": "eng_modals_deduction",
            "topic": "Модальные глаголы",
            "score": 0.4,
            "completed_at": "2026-07-21T10:00:00+00:00",
        },
    ]

    plan = build_review_plan(records, datetime(2026, 7, 22, 12, tzinfo=UTC))

    assert len(plan) == 2
    assert plan[0]["topic_key"] == "eng_modals_deduction"
    assert plan[0]["status"] == "сегодня"
    assert plan[1]["topic_key"] == "eng_articles"
    assert plan[1]["interval_days"] == 7
