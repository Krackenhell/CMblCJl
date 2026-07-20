from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

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
        ProbeQuestion(
            id="leakage-counterfactual",
            skill_id="data_leakage",
            text=(
                "Допустим, масштабирование уже выполнено на всём наборе данных. "
                "Какая информация из тестовой части стала доступна модели косвенно?"
            ),
            purpose="Проверить понимание механизма утечки, а не знание готового правила.",
            expected_concepts=(
                ("средн", "статист"),
                ("отклон", "масштаб"),
                ("тест", "test"),
            ),
            misconception_patterns={
                "scaling_is_always_safe": ("никакая", "не влияет", "безопасно"),
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
        ProbeQuestion(
            id="metrics-cost",
            skill_id="metrics",
            text=(
                "В задаче пропуск положительного случая намного дороже ложной тревоги. "
                "Какую метрику нужно контролировать в первую очередь и почему?"
            ),
            purpose="Проверить связь метрики со стоимостью ошибки.",
            expected_concepts=(
                ("recall", "полнот"),
                ("ложноотриц", "false negative", "пропуск"),
                ("цен", "дороже", "важн"),
            ),
            misconception_patterns={
                "accuracy_is_universal": ("только accuracy", "accuracy достаточно"),
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
        ProbeQuestion(
            id="reproducibility-audit",
            skill_id="reproducibility",
            text=(
                "Два запуска дали разные результаты. Назови минимум три источника "
                "случайности или изменения среды, которые нужно проверить."
            ),
            purpose="Проверить практическую диагностику невоспроизводимости.",
            expected_concepts=(
                ("разбиен", "random_state", "seed", "сид"),
                ("инициализац", "модел"),
                ("верс", "данн", "библиот"),
            ),
            misconception_patterns={
                "deterministic_by_default": ("источников нет", "всегда одинаков"),
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
        ProbeQuestion(
            id="split-model-choice",
            skill_id="validation_split",
            text=(
                "У тебя есть три модели. На какой выборке ты выберешь лучшую, "
                "а на какой только один раз сообщишь итоговое качество?"
            ),
            purpose="Проверить применение ролей выборок в выборе модели.",
            expected_concepts=(
                ("валидац", "validation"),
                ("тест", "test"),
                ("один раз", "финаль", "независ"),
            ),
            misconception_patterns={
                "test_is_validation": ("выберу по test", "тест для выбора"),
            },
        ),
    ),
    "cross_validation": (
        ProbeQuestion(
            id="cv-purpose",
            skill_id="cross_validation",
            text=(
                "Зачем использовать перекрёстную проверку вместо одного случайного "
                "разбиения и почему тестовую выборку всё равно нужно оставить отдельно?"
            ),
            purpose="Проверить назначение перекрёстной проверки и независимого теста.",
            expected_concepts=(
                ("несколько", "фолд", "разбиен"),
                ("устойчив", "разброс", "оцен"),
                ("тест", "независ", "финаль"),
            ),
            misconception_patterns={
                "cv_replaces_test": ("тест не нужен", "заменяет test"),
            },
        ),
        ProbeQuestion(
            id="cv-folds",
            skill_id="cross_validation",
            text=(
                "Что происходит с одной строкой данных во время k-fold проверки: "
                "сколько раз она участвует в обучении и сколько — в проверке?"
            ),
            purpose="Проверить понимание механики k-fold.",
            expected_concepts=(
                ("k-1", "всех кроме одного", "несколько раз"),
                ("один раз", "одном фолде"),
                ("валидац", "проверк"),
            ),
            misconception_patterns={
                "cv_same_split": ("одно разбиение", "всегда одна"),
            },
        ),
    ),
}


def select_questions(
    findings: list[ArtifactFinding],
    mastery: dict[str, float],
    limit: int = 3,
    allowed_skills: list[str] | None = None,
    seed_key: str = "",
) -> list[ProbeQuestion]:
    allowed = set(allowed_skills or QUESTION_BANK)
    priorities: dict[str, float] = defaultdict(float)
    for finding in findings:
        if finding.skill_id not in allowed:
            continue
        severity_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(finding.severity, 0.4)
        priorities[finding.skill_id] += severity_weight * finding.confidence

    for skill_id in allowed:
        if skill_id not in QUESTION_BANK:
            continue
        priorities[skill_id] += 1 - mastery.get(skill_id, 0.35)

    ordered_skills = sorted(priorities, key=priorities.get, reverse=True)
    selected: list[ProbeQuestion] = []
    for skill_id in ordered_skills:
        if QUESTION_BANK.get(skill_id):
            variants = QUESTION_BANK[skill_id]
            digest = sha256(f"{seed_key}:{skill_id}".encode()).digest()
            selected.append(variants[digest[0] % len(variants)])
        if len(selected) == limit:
            break
    return selected


def follow_up_question(question: ProbeQuestion, seed_key: str = "") -> ProbeQuestion | None:
    variants = [item for item in QUESTION_BANK.get(question.skill_id, ()) if item.id != question.id]
    if not variants:
        return None
    digest = sha256(f"follow-up:{seed_key}:{question.id}".encode()).digest()
    return variants[digest[0] % len(variants)]
