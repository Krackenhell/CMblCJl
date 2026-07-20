from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParameters:
    """Parameters of an interpretable Bayesian Knowledge Tracing model.

    The four-parameter structure follows standard BKT and the design used by
    OATutor. VivaTrace adds fractional evidence so a rubric score can represent
    partial understanding instead of collapsing every viva answer to correct/
    incorrect.
    """

    prior: float = 0.35
    learn: float = 0.12
    slip: float = 0.10
    guess: float = 0.20

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


class BKTModel:
    """Updates skill mastery from continuous evidence in the [0, 1] range."""

    def __init__(self, params: BKTParameters | None = None) -> None:
        self.params = params or BKTParameters()

    def update(self, mastery: float, evidence_score: float) -> float:
        mastery = min(max(mastery, 1e-6), 1 - 1e-6)
        score = min(max(evidence_score, 0.0), 1.0)

        # A fractional score interpolates between correct and incorrect
        # likelihoods. This is useful for open oral answers graded by a rubric.
        p_evidence_known = score * (1 - self.params.slip) + (1 - score) * self.params.slip
        p_evidence_unknown = score * self.params.guess + (1 - score) * (1 - self.params.guess)

        numerator = mastery * p_evidence_known
        denominator = numerator + (1 - mastery) * p_evidence_unknown
        posterior = numerator / denominator if denominator else mastery
        learned = posterior + (1 - posterior) * self.params.learn
        return round(min(max(learned, 0.0), 1.0), 4)

    def update_many(self, mastery: float, scores: list[float]) -> float:
        current = mastery
        for score in scores:
            current = self.update(current, score)
        return current


def combine_mastery_evidence(
    previous_mastery: float,
    submission_score: float,
    viva_scores: list[float],
    *,
    history_weight: float = 0.20,
) -> float:
    """Fuse historical, task and Viva evidence without letting one probe erase the task.

    The current cycle gives equal weight to the submitted work and the mean of
    all Viva answers. Historical mastery smooths the result across attempts.
    """
    previous = min(max(previous_mastery, 0.0), 1.0)
    submission = min(max(submission_score, 0.0), 1.0)
    if viva_scores:
        viva = sum(min(max(score, 0.0), 1.0) for score in viva_scores) / len(viva_scores)
        cycle = 0.5 * submission + 0.5 * viva
    else:
        cycle = submission
    combined = history_weight * previous + (1 - history_weight) * cycle
    return round(min(max(combined, 0.0), 1.0), 4)
