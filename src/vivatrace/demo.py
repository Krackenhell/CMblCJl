from __future__ import annotations

import json
from pathlib import Path

from .models import Evidence, StudentState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_demo_students() -> list[StudentState]:
    payload = json.loads((DATA_DIR / "demo_students.json").read_text(encoding="utf-8"))
    students: list[StudentState] = []
    for item in payload:
        evidence = [Evidence(**entry) for entry in item.get("evidence", [])]
        students.append(
            StudentState(
                student_id=item["student_id"],
                name=item["name"],
                mastery=item["mastery"],
                evidence=evidence,
                metadata=item.get("metadata", {}),
            )
        )
    return students


DEMO_ASSIGNMENT = """Задача: построить baseline-модель бинарной классификации,
оценить качество и описать схему валидации. Обоснуйте выбор метрик."""


DEMO_SUBMISSION = """from sklearn.model_selection import train_test_split
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
"""

