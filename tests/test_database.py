from __future__ import annotations

from vivatrace.database import (
    get_mastery,
    init_database,
    latest_attempts,
    list_assignments,
    list_students,
    reset_learning_data,
    save_attempt,
    student_attempts,
)
from vivatrace.models import ArtifactFinding, Evidence


def test_database_starts_with_three_students_and_no_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv("VIVATRACE_DB_PATH", str(tmp_path / "test.db"))

    init_database()

    assert [student["id"] for student in list_students()] == ["s01", "s02", "s03"]
    assignment = list_assignments()[0]
    assert latest_attempts(assignment["id"]) == []


def test_attempt_updates_mastery_and_teacher_gets_latest_per_student(tmp_path, monkeypatch):
    monkeypatch.setenv("VIVATRACE_DB_PATH", str(tmp_path / "test.db"))
    init_database()
    assignment = list_assignments()[0]
    finding = ArtifactFinding(
        skill_id="data_leakage",
        evidence="Scaler обучен до разбиения.",
        hypothesis="Возможна утечка данных.",
        severity="high",
        confidence=0.95,
    )
    evidence = Evidence(
        skill_id="data_leakage",
        score=0.8,
        confidence=0.9,
        quote="fit нужно выполнять только на train",
        rationale="Студент верно описал источник утечки.",
        misconception=None,
    )

    save_attempt("s01", assignment["id"], "first", [finding], [evidence], {"data_leakage": 0.7})
    save_attempt("s01", assignment["id"], "second", [], [evidence], {"data_leakage": 0.82})
    save_attempt("s02", assignment["id"], "third", [finding], [evidence], {"data_leakage": 0.61})

    teacher_rows = latest_attempts(assignment["id"])
    assert len(teacher_rows) == 2
    assert {row["artifact"] for row in teacher_rows} == {"second", "third"}
    assert len(student_attempts("s01", assignment["id"])) == 2
    assert get_mastery("s01", ["data_leakage"])["data_leakage"] == 0.82

    reset_learning_data()
    assert latest_attempts(assignment["id"]) == []
    assert get_mastery("s01", ["data_leakage"])["data_leakage"] == 0.35
