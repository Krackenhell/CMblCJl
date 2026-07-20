from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ArtifactFinding, Evidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "vivatrace.db"
STUDENT_SEED = (
    ("s01", "Анна Морозова"),
    ("s02", "Максим Волков"),
    ("s03", "Дарья Соколова"),
)

DEFAULT_ASSIGNMENT = {
    "title": "Проверка модели без утечки данных",
    "topic": "Валидация модели",
    "instructions": (
        "Постройте базовую модель бинарной классификации, разделите данные, "
        "выполните предобработку и обоснуйте выбор метрик. Вставьте код решения ниже."
    ),
    "starter_code": """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df.drop(columns=["target"])
y = df["target"]
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, pred))
""",
    "skill_ids": [
        "validation_split",
        "data_leakage",
        "metrics",
        "cross_validation",
        "reproducibility",
    ],
}


def database_path() -> Path:
    configured = os.getenv("VIVATRACE_DB_PATH")
    return Path(configured) if configured else DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(database_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                instructions TEXT NOT NULL,
                starter_code TEXT NOT NULL DEFAULT '',
                skill_ids_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(id),
                assignment_id INTEGER NOT NULL REFERENCES assignments(id),
                artifact TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                mastery_json TEXT NOT NULL,
                overall_score REAL NOT NULL,
                completed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mastery (
                student_id TEXT NOT NULL REFERENCES students(id),
                skill_id TEXT NOT NULL,
                value REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (student_id, skill_id)
            );
            """
        )
        connection.executemany(
            "INSERT OR IGNORE INTO students(id, name) VALUES (?, ?)",
            STUDENT_SEED,
        )
        assignment_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        if assignment_count == 0:
            create_assignment(connection=connection, **DEFAULT_ASSIGNMENT)


def list_students() -> list[dict[str, str]]:
    with connect() as connection:
        rows = connection.execute("SELECT id, name FROM students ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def list_assignments(active_only: bool = True) -> list[dict[str, Any]]:
    query = "SELECT * FROM assignments"
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY id DESC"
    with connect() as connection:
        rows = connection.execute(query).fetchall()
    return [_assignment_from_row(row) for row in rows]


def get_assignment(assignment_id: int) -> dict[str, Any]:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM assignments WHERE id = ?", (assignment_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Assignment {assignment_id} not found")
    return _assignment_from_row(row)


def _assignment_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["skill_ids"] = json.loads(item.pop("skill_ids_json"))
    item["active"] = bool(item["active"])
    return item


def create_assignment(
    title: str,
    topic: str,
    instructions: str,
    starter_code: str,
    skill_ids: list[str],
    connection: sqlite3.Connection | None = None,
) -> int:
    owns_connection = connection is None
    connection = connection or connect()
    cursor = connection.execute(
        """
        INSERT INTO assignments(title, topic, instructions, starter_code, skill_ids_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            title.strip(),
            topic.strip(),
            instructions.strip(),
            starter_code,
            json.dumps(skill_ids, ensure_ascii=False),
            datetime.now(UTC).isoformat(),
        ),
    )
    if owns_connection:
        connection.commit()
        connection.close()
    return int(cursor.lastrowid)


def update_assignment(
    assignment_id: int,
    title: str,
    topic: str,
    instructions: str,
    starter_code: str,
    skill_ids: list[str],
) -> None:
    with connect() as connection:
        connection.execute(
            """
            UPDATE assignments
            SET title = ?, topic = ?, instructions = ?, starter_code = ?, skill_ids_json = ?
            WHERE id = ?
            """,
            (
                title.strip(),
                topic.strip(),
                instructions.strip(),
                starter_code,
                json.dumps(skill_ids, ensure_ascii=False),
                assignment_id,
            ),
        )


def get_mastery(student_id: str, skill_ids: list[str], prior: float = 0.35) -> dict[str, float]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT skill_id, value FROM mastery WHERE student_id = ?", (student_id,)
        ).fetchall()
    stored = {row["skill_id"]: float(row["value"]) for row in rows}
    return {skill_id: stored.get(skill_id, prior) for skill_id in skill_ids}


def save_attempt(
    student_id: str,
    assignment_id: int,
    artifact: str,
    findings: list[ArtifactFinding],
    evidence: list[Evidence],
    mastery: dict[str, float],
) -> int:
    now = datetime.now(UTC).isoformat()
    overall_score = sum(item.score for item in evidence) / max(len(evidence), 1)
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO attempts(
                student_id, assignment_id, artifact, findings_json, evidence_json,
                mastery_json, overall_score, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                assignment_id,
                artifact,
                json.dumps([asdict(item) for item in findings], ensure_ascii=False),
                json.dumps([asdict(item) for item in evidence], ensure_ascii=False),
                json.dumps(mastery, ensure_ascii=False),
                overall_score,
                now,
            ),
        )
        connection.executemany(
            """
            INSERT INTO mastery(student_id, skill_id, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id, skill_id)
            DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            [(student_id, skill_id, value, now) for skill_id, value in mastery.items()],
        )
    return int(cursor.lastrowid)


def latest_attempts(assignment_id: int) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT a.*, s.name AS student_name
            FROM attempts a
            JOIN students s ON s.id = a.student_id
            JOIN (
                SELECT student_id, MAX(id) AS latest_id
                FROM attempts
                WHERE assignment_id = ?
                GROUP BY student_id
            ) latest ON latest.latest_id = a.id
            ORDER BY s.name
            """,
            (assignment_id,),
        ).fetchall()
    return [_attempt_from_row(row) for row in rows]


def student_attempts(student_id: str, assignment_id: int) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM attempts
            WHERE student_id = ? AND assignment_id = ?
            ORDER BY id DESC
            """,
            (student_id, assignment_id),
        ).fetchall()
    return [_attempt_from_row(row) for row in rows]


def _attempt_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["findings"] = json.loads(item.pop("findings_json"))
    item["evidence"] = json.loads(item.pop("evidence_json"))
    item["mastery"] = json.loads(item.pop("mastery_json"))
    return item


def reset_learning_data() -> None:
    with connect() as connection:
        connection.execute("DELETE FROM attempts")
        connection.execute("DELETE FROM mastery")
