from __future__ import annotations

import re

from .models import ArtifactFinding


def inspect_ml_artifact(text: str) -> list[ArtifactFinding]:
    """Find teachable signals in a small ML assignment.

    This is deliberately an evidence extractor, not an automatic accusation or
    final grader. Findings become hypotheses that the viva agent must verify.
    """

    normalized = " ".join(text.lower().split())
    findings: list[ArtifactFinding] = []

    fit_positions = [match.start() for match in re.finditer(r"fit(?:_transform)?\s*\(", normalized)]
    split_calls = [match.start() for match in re.finditer(r"train_test_split\s*\(", normalized)]
    split_position = min(split_calls) if split_calls else -1
    if split_position >= 0 and fit_positions and min(fit_positions) < split_position:
        findings.append(
            ArtifactFinding(
                skill_id="data_leakage",
                severity="high",
                evidence="Преобразование fit/fit_transform выполнено до train_test_split.",
                hypothesis="Статистики тестовой выборки могли попасть в обучение.",
                confidence=0.93,
            )
        )

    if "accuracy" in normalized and not any(
        metric in normalized for metric in ("f1", "precision", "recall", "roc_auc")
    ):
        findings.append(
            ArtifactFinding(
                skill_id="metrics",
                severity="medium",
                evidence="В отчёте используется только accuracy.",
                hypothesis="Нужно проверить понимание метрик при дисбалансе классов.",
                confidence=0.78,
            )
        )

    if "train_test_split" in normalized and "random_state" not in normalized:
        findings.append(
            ArtifactFinding(
                skill_id="reproducibility",
                severity="medium",
                evidence="Разбиение не фиксирует random_state.",
                hypothesis="Результат эксперимента может быть невоспроизводимым.",
                confidence=0.86,
            )
        )

    if "test" in normalized and any(term in normalized for term in ("gridsearch", "optuna", "best_param")):
        findings.append(
            ArtifactFinding(
                skill_id="validation_split",
                severity="high",
                evidence="В работе одновременно упоминаются test set и подбор гиперпараметров.",
                hypothesis="Нужно проверить, не использовался ли test set для выбора модели.",
                confidence=0.70,
            )
        )

    return findings
