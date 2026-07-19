from __future__ import annotations

from collections import Counter

import pandas as pd

from .models import Curriculum, StudentState


def mastery_frame(curriculum: Curriculum, students: list[StudentState]) -> pd.DataFrame:
    rows = []
    for student in students:
        row = {"student_id": student.student_id, "student": student.name}
        row.update({skill.id: student.mastery.get(skill.id, 0.35) for skill in curriculum.skills})
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_skill_summary(curriculum: Curriculum, students: list[StudentState]) -> pd.DataFrame:
    frame = mastery_frame(curriculum, students)
    records = []
    for skill in curriculum.skills:
        values = frame[skill.id]
        records.append(
            {
                "skill_id": skill.id,
                "Навык": skill.name,
                "Среднее освоение": float(values.mean()),
                "Нужна помощь": int((values < 0.45).sum()),
                "Нужно закрепить": int(((values >= 0.45) & (values < skill.target_mastery)).sum()),
                "Готовы к усложнению": int((values >= skill.target_mastery).sum()),
                "Доля с пробелом": float((values < 0.6).mean()),
            }
        )
    return pd.DataFrame(records).sort_values("Доля с пробелом", ascending=False)


def misconception_summary(students: list[StudentState]) -> pd.DataFrame:
    counts: Counter[str] = Counter(
        evidence.misconception
        for student in students
        for evidence in student.evidence
        if evidence.misconception
    )
    labels = {
        "scaling_is_always_safe": "Масштабирование до split считается безопасным",
        "test_can_train_preprocessing": "Test set используется для обучения preprocessing",
        "accuracy_is_universal": "Accuracy считается универсальной метрикой",
        "deterministic_by_default": "Эксперимент считается воспроизводимым по умолчанию",
        "test_is_validation": "Test и validation смешиваются",
    }
    rows = [
        {"Заблуждение": labels.get(key, key), "Студентов": value}
        for key, value in counts.most_common()
    ]
    return pd.DataFrame(rows, columns=["Заблуждение", "Студентов"])


def intervention_for_gap(gap_share: float, skill_name: str) -> dict[str, str]:
    if gap_share >= 0.35:
        return {
            "level": "Вся группа",
            "decision": f"Начать следующую пару с 15-минутного разбора «{skill_name}».",
            "format": "Контрпример → парное обсуждение → один exit-ticket.",
        }
    if gap_share >= 0.10:
        return {
            "level": "Малая группа",
            "decision": f"Назначить адресный практикум по навыку «{skill_name}».",
            "format": "Один разобранный пример → одно самостоятельное исправление.",
        }
    return {
        "level": "Индивидуально",
        "decision": f"Сохранить общий темп и выдать персональный repair по «{skill_name}».",
        "format": "Микрообъяснение → повторный viva-вопрос.",
    }

