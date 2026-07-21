from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from vivatrace.bkt import BKTModel, BKTParameters


ROOT = Path(__file__).resolve().parents[1]


def calibration_error(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return error


def metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    eps = 1e-6
    probabilities = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    labels = np.asarray(y_true, dtype=float)
    return {
        "brier": round(float(np.mean((probabilities - labels) ** 2)), 6),
        "log_loss": round(float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))), 6),
        "ece_10": round(calibration_error(y_true, y_prob, 10), 6),
        "accuracy_0_5": round(float(np.mean((probabilities >= 0.5) == labels)), 6),
    }


def generate_synthetic_pilot(seed: int = 42, students: int = 180, steps: int = 12) -> list[dict]:
    """Transparent pilot data for pipeline validation before real cohort collection."""
    rng = np.random.default_rng(seed)
    rows = []
    for student_index in range(students):
        latent = float(rng.beta(2.2, 3.2))
        learning_rate = float(rng.uniform(0.025, 0.085))
        sequence = []
        for step in range(steps + 1):
            noisy_score = float(np.clip(latent + rng.normal(0, 0.14), 0, 1))
            outcome = int(rng.random() < latent)
            sequence.append((noisy_score, outcome))
            latent = float(np.clip(latent + learning_rate * (1 - latent), 0, 0.98))
        rows.append(
            {
                "student_id": f"synthetic-{student_index:03d}",
                "events": [
                    {
                        "evidence_score": sequence[index][0],
                        "next_success": sequence[index + 1][1],
                    }
                    for index in range(steps)
                ],
            }
        )
    return rows


def predict_bkt(sequences: list[dict], params: BKTParameters) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    predictions: list[float] = []
    model = BKTModel(params)
    for sequence in sequences:
        mastery = params.prior
        for event in sequence["events"]:
            mastery = model.update(mastery, float(event["evidence_score"]))
            predictions.append(mastery)
            labels.append(int(event["next_success"]))
    return labels, predictions


def predict_latest(sequences: list[dict]) -> tuple[list[int], list[float]]:
    labels, predictions = [], []
    for sequence in sequences:
        for event in sequence["events"]:
            labels.append(int(event["next_success"]))
            predictions.append(float(event["evidence_score"]))
    return labels, predictions


def predict_ema(sequences: list[dict], alpha: float, prior: float) -> tuple[list[int], list[float]]:
    labels, predictions = [], []
    for sequence in sequences:
        state = prior
        for event in sequence["events"]:
            state = alpha * float(event["evidence_score"]) + (1 - alpha) * state
            labels.append(int(event["next_success"]))
            predictions.append(state)
    return labels, predictions


def fit_bkt(train: list[dict]) -> BKTParameters:
    candidates = product(
        [0.25, 0.35, 0.45],
        [0.03, 0.07, 0.12],
        [0.08, 0.14, 0.20],
        [0.18, 0.25, 0.32],
    )
    best_params, best_brier = None, math.inf
    for prior, learn, slip, guess in candidates:
        params = BKTParameters(prior=prior, learn=learn, slip=slip, guess=guess)
        labels, predictions = predict_bkt(train, params)
        brier = metrics(labels, predictions)["brier"]
        if brier < best_brier:
            best_params, best_brier = params, brier
    assert best_params is not None
    return best_params


def run_experiment(seed: int = 42) -> dict:
    sequences = generate_synthetic_pilot(seed=seed)
    split = int(len(sequences) * 0.7)
    train, test = sequences[:split], sequences[split:]
    calibration_size = max(1, int(len(train) * 0.2))
    fit_sequences, calibration_sequences = train[:-calibration_size], train[-calibration_size:]
    train_labels, _ = predict_latest(fit_sequences)
    global_prior = sum(train_labels) / len(train_labels)
    best_alpha = min(
        [0.15, 0.30, 0.50, 0.70],
        key=lambda alpha: metrics(*predict_ema(fit_sequences, alpha, global_prior))["brier"],
    )
    fitted_bkt = fit_bkt(fit_sequences)
    calibration_labels, calibration_predictions = predict_bkt(
        calibration_sequences, fitted_bkt
    )
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(calibration_predictions, calibration_labels)

    labels, latest_predictions = predict_latest(test)
    _, ema_predictions = predict_ema(test, best_alpha, global_prior)
    _, bkt_predictions = predict_bkt(test, fitted_bkt)
    bkt_calibrated = [float(item) for item in isotonic.predict(bkt_predictions)]
    global_predictions = [global_prior] * len(labels)
    results = {
        "global_mean": metrics(labels, global_predictions),
        "latest_evidence": metrics(labels, latest_predictions),
        "ema": metrics(labels, ema_predictions),
        "bkt": metrics(labels, bkt_predictions),
        "bkt_isotonic": metrics(labels, bkt_calibrated),
    }
    return {
        "experiment": "knowledge_tracing_calibration_v1",
        "dataset": {
            "type": "synthetic_pilot",
            "warning": "Проверяет экспериментальный пайплайн; для продуктовых выводов нужны реальные последовательности студентов.",
            "seed": seed,
            "students": len(sequences),
            "train_students": len(train),
            "fit_students": len(fit_sequences),
            "calibration_students": len(calibration_sequences),
            "test_students": len(test),
            "test_events": len(labels),
        },
        "selection_metric": "Brier score on train",
        "fitted": {"ema_alpha": best_alpha, "bkt": asdict(fitted_bkt)},
        "test_metrics": results,
        "best_by_brier": min(results, key=lambda name: results[name]["brier"]),
        "best_by_calibration": min(results, key=lambda name: results[name]["ece_10"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment(args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
