from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experiment_metrics.json"


def make_dataset(n: int = 800, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a bounded failure mode: polished work without mastery.

    This synthetic experiment tests whether the pipeline can exploit an
    additional knowledge_check signal. It is not presented as evidence of real-world
    educational impact.
    """

    rng = np.random.default_rng(seed)
    latent_mastery = rng.beta(2.2, 2.0, size=n)
    true_mastery = (latent_mastery >= 0.58).astype(int)

    assistance = rng.binomial(1, 0.28, size=n)
    assignment_noise = rng.normal(0, 0.16, size=n)
    assignment_score = np.clip(latent_mastery + 0.42 * assistance + assignment_noise, 0, 1)

    knowledge_check_noise = rng.normal(0, 0.11, size=n)
    knowledge_check_score = np.clip(latent_mastery + knowledge_check_noise - 0.03 * assistance, 0, 1)
    artifact_flags = np.clip(1.1 - latent_mastery + rng.normal(0, 0.18, size=n), 0, 1)

    assignment_only = assignment_score.reshape(-1, 1)
    hybrid = np.column_stack([assignment_score, knowledge_check_score, artifact_flags])
    return assignment_only, hybrid, true_mastery


def evaluate(features: np.ndarray, target: np.ndarray, seed: int = 42) -> dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, random_state=seed, stratify=target
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, prediction)), 4),
        "f1": round(float(f1_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
    }


def main() -> None:
    assignment_only, hybrid, target = make_dataset()
    payload = {
        "dataset": {
            "type": "synthetic proof-of-pipeline",
            "samples": int(len(target)),
            "seed": 42,
            "positive_share": round(float(target.mean()), 4),
            "assumption": (
                "28% submissions receive an external polish boost that improves the artifact "
                "score without changing latent mastery."
            ),
            "limitation": "Not evidence of learning impact; requires a real student pilot.",
        },
        "models": {
            "assignment_only": evaluate(assignment_only, target),
            "assignment_plus_knowledge_check": evaluate(hybrid, target),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

