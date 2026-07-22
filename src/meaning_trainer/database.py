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
DEFAULT_DB_PATH = PROJECT_ROOT / "meaning_trainer.db"
ENGLISH_ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "english_b2_assignments.json"
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
    "subject": "Машинное обучение",
    "topic_key": "ml_validation",
    "difficulty": 2,
    "variant": 1,
    "rubric": {
        "reference_answer": "Разделение выполняется до обучения преобразований; preprocessing обучается только на train через Pipeline; фиксируется random_state; помимо accuracy рассматриваются метрики под дисбаланс; test не используется для подбора модели.",
        "criteria": [
            "нет утечки данных из test",
            "разделены роли train/validation/test",
            "метрики соответствуют задаче",
            "эксперимент воспроизводим",
        ],
        "common_errors": [
            "fit_transform до train_test_split",
            "только accuracy",
            "нет random_state",
        ],
    },
}


def database_path() -> Path:
    configured = os.getenv("MEANING_DB_PATH")
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

            CREATE TABLE IF NOT EXISTS mission_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(id),
                mission_id TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                turn_count INTEGER NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 0,
                state_json TEXT NOT NULL DEFAULT '{}',
                messages_json TEXT NOT NULL DEFAULT '[]',
                traces_json TEXT NOT NULL DEFAULT '[]',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_mission_attempts_student
            ON mission_attempts(student_id, mission_id, id DESC);

            CREATE TABLE IF NOT EXISTS voice_sessions (
                session_id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL REFERENCES students(id),
                assignment_id INTEGER NOT NULL REFERENCES assignments(id),
                status TEXT NOT NULL DEFAULT 'active',
                turn_count INTEGER NOT NULL DEFAULT 0,
                average_score REAL NOT NULL DEFAULT 0,
                average_fluency REAL NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS voice_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES voice_sessions(session_id),
                turn_index INTEGER NOT NULL,
                student_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                assessment_json TEXT NOT NULL DEFAULT '{}',
                trace_json TEXT NOT NULL DEFAULT '{}',
                overall_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, turn_index)
            );

            CREATE INDEX IF NOT EXISTS idx_voice_sessions_student
            ON voice_sessions(student_id, assignment_id, updated_at DESC);
            """
        )
        _migrate_schema(connection)
        connection.executemany(
            "INSERT OR IGNORE INTO students(id, name) VALUES (?, ?)",
            STUDENT_SEED,
        )
        assignment_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        if assignment_count == 0:
            create_assignment(connection=connection, **DEFAULT_ASSIGNMENT)
        _seed_english_assignments(connection)


def _migrate_schema(connection: sqlite3.Connection) -> None:
    assignment_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(assignments)").fetchall()
    }
    attempt_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(attempts)").fetchall()
    }
    assignment_additions = {
        "subject": "TEXT NOT NULL DEFAULT 'Машинное обучение'",
        "rubric_json": "TEXT NOT NULL DEFAULT '{}'",
        "topic_key": "TEXT NOT NULL DEFAULT ''",
        "difficulty": "INTEGER NOT NULL DEFAULT 1",
        "variant": "INTEGER NOT NULL DEFAULT 1",
    }
    attempt_additions = {
        "submission_score": "REAL",
        "submission_correct": "INTEGER",
        "assessment_mode": "TEXT",
        "next_activity_json": "TEXT NOT NULL DEFAULT '{}'",
        "teacher_recommendation_json": "TEXT NOT NULL DEFAULT '{}'",
        "traces_json": "TEXT NOT NULL DEFAULT '[]'",
        "assessment_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for column, definition in assignment_additions.items():
        if column not in assignment_columns:
            connection.execute(f"ALTER TABLE assignments ADD COLUMN {column} {definition}")
    for column, definition in attempt_additions.items():
        if column not in attempt_columns:
            connection.execute(f"ALTER TABLE attempts ADD COLUMN {column} {definition}")
    connection.execute(
        """
        UPDATE assignments
        SET topic_key = COALESCE(json_extract(skill_ids_json, '$[0]'), 'general')
        WHERE topic_key = ''
        """
    )


def _seed_english_assignments(connection: sqlite3.Connection) -> None:
    payload = json.loads(ENGLISH_ASSIGNMENTS_PATH.read_text(encoding="utf-8"))
    existing = {
        row["title"]: dict(row)
        for row in connection.execute(
            "SELECT id, title, topic_key FROM assignments"
        ).fetchall()
    }
    for assignment in payload:
        current = existing.get(assignment["title"])
        if current is None:
            create_assignment(connection=connection, **assignment)
        elif not current.get("topic_key"):
            connection.execute(
                """
                UPDATE assignments
                SET instructions = ?, starter_code = ?, rubric_json = ?, topic_key = ?,
                    difficulty = ?, variant = ?
                WHERE id = ?
                """,
                (
                    assignment["instructions"],
                    assignment["starter_code"],
                    json.dumps(assignment["rubric"], ensure_ascii=False),
                    assignment["topic_key"],
                    assignment["difficulty"],
                    assignment["variant"],
                    current["id"],
                ),
            )


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
    item["rubric"] = json.loads(item.pop("rubric_json", "{}"))
    return item


def create_assignment(
    title: str,
    topic: str,
    instructions: str,
    starter_code: str,
    skill_ids: list[str],
    subject: str = "Учебный курс",
    rubric: dict[str, Any] | None = None,
    topic_key: str = "",
    difficulty: int = 1,
    variant: int = 1,
    connection: sqlite3.Connection | None = None,
) -> int:
    owns_connection = connection is None
    connection = connection or connect()
    cursor = connection.execute(
        """
        INSERT INTO assignments(
            title, topic, instructions, starter_code, skill_ids_json, subject, rubric_json,
            topic_key, difficulty, variant, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            title.strip(),
            topic.strip(),
            instructions.strip(),
            starter_code,
            json.dumps(skill_ids, ensure_ascii=False),
            subject.strip(),
            json.dumps(rubric or {}, ensure_ascii=False),
            topic_key.strip() or skill_ids[0],
            max(1, min(int(difficulty), 3)),
            max(1, int(variant)),
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
    subject: str = "Учебный курс",
    rubric: dict[str, Any] | None = None,
    topic_key: str = "",
    difficulty: int = 1,
    variant: int = 1,
) -> None:
    with connect() as connection:
        connection.execute(
            """
            UPDATE assignments
            SET title = ?, topic = ?, instructions = ?, starter_code = ?, skill_ids_json = ?,
                subject = ?, rubric_json = ?
                , topic_key = ?, difficulty = ?, variant = ?
            WHERE id = ?
            """,
            (
                title.strip(),
                topic.strip(),
                instructions.strip(),
                starter_code,
                json.dumps(skill_ids, ensure_ascii=False),
                subject.strip(),
                json.dumps(rubric or {}, ensure_ascii=False),
                topic_key.strip() or skill_ids[0],
                max(1, min(int(difficulty), 3)),
                max(1, int(variant)),
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


def update_mastery(student_id: str, mastery: dict[str, float]) -> None:
    """Persist externally calculated mastery without creating a fake assignment attempt."""
    if not mastery:
        return
    now = datetime.now(UTC).isoformat()
    with connect() as connection:
        connection.executemany(
            """
            INSERT INTO mastery(student_id, skill_id, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id, skill_id)
            DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            [
                (student_id, skill_id, min(max(float(value), 0.0), 1.0), now)
                for skill_id, value in mastery.items()
            ],
        )


def start_mission_attempt(student_id: str, mission: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    messages = [
        {
            "role": "npc",
            "content": str(mission["opening"]),
            "created_at": now,
        }
    ]
    state = {
        "mastery_applied": False,
        "signal": {"coverage": 0, "features": [], "found": [], "missing": []},
    }
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO mission_attempts(
                student_id, mission_id, topic_key, skill_id, title, status,
                turn_count, score, state_json, messages_json, traces_json,
                started_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', 0, 0, ?, ?, '[]', ?, ?)
            """,
            (
                student_id,
                mission["id"],
                mission["topic_key"],
                mission["skill_id"],
                mission["title"],
                json.dumps(state, ensure_ascii=False),
                json.dumps(messages, ensure_ascii=False),
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT * FROM mission_attempts WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return _mission_attempt_from_row(row)


def save_mission_turn(
    attempt_id: int,
    messages: list[dict[str, Any]],
    state: dict[str, Any],
    score: float,
    status: str,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    if status not in {"active", "completed", "needs_retry"}:
        raise ValueError(f"Неизвестный статус миссии: {status}")
    now = datetime.now(UTC).isoformat()
    turn_count = sum(message.get("role") == "student" for message in messages)
    completed_at = now if status == "completed" else None
    with connect() as connection:
        connection.execute(
            """
            UPDATE mission_attempts
            SET status = ?, turn_count = ?, score = ?, state_json = ?,
                messages_json = ?, traces_json = ?, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status,
                turn_count,
                min(max(float(score), 0.0), 1.0),
                json.dumps(state, ensure_ascii=False),
                json.dumps(messages, ensure_ascii=False),
                json.dumps(traces, ensure_ascii=False),
                now,
                completed_at,
                attempt_id,
            ),
        )
        row = connection.execute(
            "SELECT * FROM mission_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Mission attempt {attempt_id} not found")
    return _mission_attempt_from_row(row)


def latest_mission_attempt(student_id: str, mission_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT * FROM mission_attempts
            WHERE student_id = ? AND mission_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (student_id, mission_id),
        ).fetchone()
    return _mission_attempt_from_row(row) if row else None


def mission_history(
    student_id: str | None = None, topic_key: str | None = None
) -> list[dict[str, Any]]:
    filters: list[str] = []
    parameters: list[str] = []
    if student_id:
        filters.append("m.student_id = ?")
        parameters.append(student_id)
    if topic_key:
        filters.append("m.topic_key = ?")
        parameters.append(topic_key)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT m.*, s.name AS student_name
            FROM mission_attempts m
            JOIN students s ON s.id = m.student_id
            {where}
            ORDER BY m.id DESC
            """,
            parameters,
        ).fetchall()
    return [_mission_attempt_from_row(row) for row in rows]


def _mission_attempt_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["state"] = json.loads(item.pop("state_json") or "{}")
    item["messages"] = json.loads(item.pop("messages_json") or "[]")
    item["traces"] = json.loads(item.pop("traces_json") or "[]")
    return item


def save_attempt(
    student_id: str,
    assignment_id: int,
    artifact: str,
    findings: list[ArtifactFinding],
    evidence: list[Evidence],
    mastery: dict[str, float],
    submission_score: float | None = None,
    submission_correct: bool | None = None,
    assessment_mode: str | None = None,
    next_activity: dict[str, Any] | None = None,
    teacher_recommendation: dict[str, Any] | None = None,
    traces: list[dict[str, Any]] | None = None,
    assessment: dict[str, Any] | None = None,
) -> int:
    now = datetime.now(UTC).isoformat()
    overall_score = sum(item.score for item in evidence) / max(len(evidence), 1)
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO attempts(
                student_id, assignment_id, artifact, findings_json, evidence_json,
                mastery_json, overall_score, completed_at, submission_score,
                submission_correct, assessment_mode, next_activity_json,
                teacher_recommendation_json, traces_json, assessment_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                submission_score,
                None if submission_correct is None else int(submission_correct),
                assessment_mode,
                json.dumps(next_activity or {}, ensure_ascii=False),
                json.dumps(teacher_recommendation or {}, ensure_ascii=False),
                json.dumps(traces or [], ensure_ascii=False),
                json.dumps(assessment or {}, ensure_ascii=False),
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


def student_progress(student_id: str) -> dict[int, dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT a.assignment_id, a.submission_score, a.overall_score, a.completed_at
            FROM attempts a
            JOIN (
                SELECT assignment_id, MAX(id) AS latest_id
                FROM attempts
                WHERE student_id = ?
                GROUP BY assignment_id
            ) latest ON latest.latest_id = a.id
            """,
            (student_id,),
        ).fetchall()
    return {int(row["assignment_id"]): dict(row) for row in rows}


def student_history(student_id: str) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT a.*, x.title AS assignment_title, x.subject, x.topic, x.topic_key
            FROM attempts a
            JOIN assignments x ON x.id = a.assignment_id
            WHERE a.student_id = ?
            ORDER BY a.id DESC
            """,
            (student_id,),
        ).fetchall()
    return [_attempt_from_row(row) for row in rows]


def latest_topic_attempts(topic_key: str) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT a.*, s.name AS student_name, x.title AS assignment_title
            FROM attempts a
            JOIN students s ON s.id = a.student_id
            JOIN assignments x ON x.id = a.assignment_id
            JOIN (
                SELECT a2.student_id, MAX(a2.id) AS latest_id
                FROM attempts a2
                JOIN assignments x2 ON x2.id = a2.assignment_id
                WHERE x2.topic_key = ?
                GROUP BY a2.student_id
            ) latest ON latest.latest_id = a.id
            ORDER BY s.name
            """,
            (topic_key,),
        ).fetchall()
    return [_attempt_from_row(row) for row in rows]


def _attempt_from_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["findings"] = json.loads(item.pop("findings_json"))
    item["evidence"] = json.loads(item.pop("evidence_json"))
    item["mastery"] = json.loads(item.pop("mastery_json"))
    item["next_activity"] = json.loads(item.pop("next_activity_json", "{}") or "{}")
    item["teacher_recommendation"] = json.loads(
        item.pop("teacher_recommendation_json", "{}") or "{}"
    )
    item["traces"] = json.loads(item.pop("traces_json", "[]") or "[]")
    item["assessment"] = json.loads(item.pop("assessment_json", "{}") or "{}")
    if item.get("submission_correct") is not None:
        item["submission_correct"] = bool(item["submission_correct"])
    return item


def reset_learning_data() -> None:
    with connect() as connection:
        connection.execute("DELETE FROM voice_turns")
        connection.execute("DELETE FROM voice_sessions")
        connection.execute("DELETE FROM mission_attempts")
        connection.execute("DELETE FROM attempts")
        connection.execute("DELETE FROM mastery")


def save_voice_turn(
    *,
    session_id: str,
    student_id: str,
    assignment_id: int,
    student_text: str,
    assistant_text: str,
    metrics: dict[str, Any],
    assessment: dict[str, Any],
    trace: dict[str, Any],
    overall_score: float,
) -> int:
    now = datetime.now(UTC).isoformat()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO voice_sessions(
                session_id, student_id, assignment_id, status, started_at, updated_at
            ) VALUES (?, ?, ?, 'active', ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (session_id, student_id, assignment_id, now, now),
        )
        turn_index = int(
            connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM voice_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        )
        cursor = connection.execute(
            """
            INSERT INTO voice_turns(
                session_id, turn_index, student_text, assistant_text, metrics_json,
                assessment_json, trace_json, overall_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                student_text,
                assistant_text,
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(assessment, ensure_ascii=False),
                json.dumps(trace, ensure_ascii=False),
                float(overall_score),
                now,
            ),
        )
        connection.execute(
            """
            UPDATE voice_sessions
            SET turn_count = (
                    SELECT COUNT(*) FROM voice_turns WHERE session_id = ?
                ),
                average_score = (
                    SELECT AVG(overall_score) FROM voice_turns WHERE session_id = ?
                ),
                average_fluency = (
                    SELECT AVG(CAST(json_extract(metrics_json, '$.fluency_score') AS REAL))
                    FROM voice_turns WHERE session_id = ?
                ),
                updated_at = ?
            WHERE session_id = ?
            """,
            (session_id, session_id, session_id, now, session_id),
        )
        return int(cursor.lastrowid)


def finish_voice_session(session_id: str) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE voice_sessions SET status = 'completed', updated_at = ? WHERE session_id = ?",
            (datetime.now(UTC).isoformat(), session_id),
        )


def latest_voice_topic_sessions(topic_key: str) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT vs.*, s.name AS student_name, a.title AS assignment_title,
                   vt.student_text AS latest_student_text,
                   vt.metrics_json AS latest_metrics_json,
                   vt.assessment_json AS latest_assessment_json
            FROM voice_sessions vs
            JOIN students s ON s.id = vs.student_id
            JOIN assignments a ON a.id = vs.assignment_id
            LEFT JOIN voice_turns vt ON vt.session_id = vs.session_id
                AND vt.turn_index = (
                    SELECT MAX(vt2.turn_index) FROM voice_turns vt2
                    WHERE vt2.session_id = vs.session_id
                )
            JOIN (
                SELECT vs2.student_id, MAX(vs2.updated_at) AS latest_at
                FROM voice_sessions vs2
                JOIN assignments a2 ON a2.id = vs2.assignment_id
                WHERE a2.topic_key = ? AND vs2.turn_count > 0
                GROUP BY vs2.student_id
            ) latest ON latest.student_id = vs.student_id
                AND latest.latest_at = vs.updated_at
            ORDER BY s.name
            """,
            (topic_key,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["latest_metrics"] = json.loads(item.pop("latest_metrics_json") or "{}")
        item["latest_assessment"] = json.loads(item.pop("latest_assessment_json") or "{}")
        result.append(item)
    return result


def student_voice_sessions(student_id: str) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT vs.*, a.title AS assignment_title, a.topic, a.topic_key
            FROM voice_sessions vs
            JOIN assignments a ON a.id = vs.assignment_id
            WHERE vs.student_id = ? AND vs.turn_count > 0
            ORDER BY vs.updated_at DESC
            """,
            (student_id,),
        ).fetchall()
    return [dict(row) for row in rows]
