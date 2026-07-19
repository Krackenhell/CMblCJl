from __future__ import annotations

from collections import defaultdict

from .models import ArtifactFinding, ProbeQuestion


QUESTION_BANK: dict[str, tuple[ProbeQuestion, ...]] = {
    "data_leakage": (
        ProbeQuestion(
            id="leakage-scaler",
            skill_id="data_leakage",
            text=(
                "В работе StandardScaler обучается до разделения данных. "
                "Почему это может завысить оценку качества и как перестроить пайплайн?"
            ),
            purpose="Проверить понимание передачи статистик test set в обучение.",
            expected_concepts=(
                ("тест", "test"),
                ("статист", "средн", "масштаб"),
                ("после раздел", "только train", "pipeline"),
            ),
            misconception_patterns={
                "scaling_is_always_safe": ("не влияет", "без разницы", "это безопасно"),
                "test_can_train_preprocessing": ("обучить на всех", "весь датасет лучше"),
            },
        ),
    ),
    "metrics": (
        ProbeQuestion(
            id="metrics-imbalance",
            skill_id="metrics",
            text=(
                "Представь, что положительный класс составляет 5%. "
                "Почему accuracy недостаточно и какие метрики ты добавишь?"
            ),
            purpose="Проверить выбор метрики под дисбаланс и цену ошибок.",
            expected_concepts=(
                ("дисбаланс", "5%", "редк"),
                ("precision", "точност"),
                ("recall", "полнот"),
                ("f1", "pr-auc", "roc-auc"),
            ),
            misconception_patterns={
                "accuracy_is_universal": ("accuracy достаточно", "всегда accuracy"),
            },
        ),
    ),
    "reproducibility": (
        ProbeQuestion(
            id="reproducibility-seed",
            skill_id="reproducibility",
            text=(
                "Что изменится при повторном запуске этого ноутбука и что нужно "
                "зафиксировать, чтобы эксперимент был воспроизводимым?"
            ),
            purpose="Проверить понимание источников случайности.",
            expected_concepts=(
                ("random_state", "seed", "сид"),
                ("разбиен", "инициализац", "случайн"),
            ),
            misconception_patterns={
                "deterministic_by_default": ("ничего не изменится", "всегда одинаков"),
            },
        ),
    ),
    "validation_split": (
        ProbeQuestion(
            id="split-purpose",
            skill_id="validation_split",
            text=(
                "Раздели роли train, validation и test. Почему нельзя подбирать "
                "гиперпараметры по test set?"
            ),
            purpose="Проверить независимость финальной оценки.",
            expected_concepts=(
                ("обуч", "train"),
                ("подбор", "validation", "валидац"),
                ("финаль", "независ", "test"),
            ),
            misconception_patterns={
                "test_is_validation": ("test для подбора", "подбирать по test"),
            },
        ),
    ),
}


def select_questions(
    findings: list[ArtifactFinding],
    mastery: dict[str, float],
    limit: int = 3,
) -> list[ProbeQuestion]:
    priorities: dict[str, float] = defaultdict(float)
    for finding in findings:
        severity_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(finding.severity, 0.4)
        priorities[finding.skill_id] += severity_weight * finding.confidence

    for skill_id in QUESTION_BANK:
        priorities[skill_id] += 1 - mastery.get(skill_id, 0.35)

    ordered_skills = sorted(priorities, key=priorities.get, reverse=True)
    selected: list[ProbeQuestion] = []
    for skill_id in ordered_skills:
        if QUESTION_BANK.get(skill_id):
            selected.append(QUESTION_BANK[skill_id][0])
        if len(selected) == limit:
            break
    return selected

