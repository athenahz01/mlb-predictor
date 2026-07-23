from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


def binary_log_loss(probabilities, outcomes, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_score(probabilities, outcomes) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probabilities, outcomes, bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean() - y[mask].mean()))
    return ece


def calibration_slope_intercept(probabilities, outcomes) -> tuple[float, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(outcomes, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logits, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def binary_metrics(probabilities, outcomes) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    slope, intercept = calibration_slope_intercept(p, y)
    return {
        "n": int(len(y)),
        "log_loss": float(binary_log_loss(p, y).mean()),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }


def date_clustered_paired_bootstrap(
    challenger_loss,
    champion_loss,
    dates,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    practical_effect: float = 0.001,
    alpha: float = 0.05,
) -> dict[str, Any]:
    challenger = np.asarray(challenger_loss, dtype=float)
    champion = np.asarray(champion_loss, dtype=float)
    clusters = np.asarray(dates)
    if not (len(challenger) == len(champion) == len(clusters)):
        raise ValueError("loss and date arrays must have equal length")
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("at least two date clusters are required")
    difference = champion - challenger
    by_cluster = {cluster: difference[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for index in range(n_boot):
        sampled_clusters = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([by_cluster[cluster] for cluster in sampled_clusters])
        samples[index] = sampled.mean()
    observed = float(difference.mean())
    quantiles: NDArray[np.float64] = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    ci_low = float(quantiles[0])
    ci_high = float(quantiles[1])
    p_value = float((np.count_nonzero(samples <= 0) + 1) / (n_boot + 1))
    return {
        "effect": observed,
        "confidence_interval": [ci_low, ci_high],
        "p_value_one_sided": p_value,
        "n_boot": n_boot,
        "seed": seed,
        "clusters": int(len(unique)),
        "practical_effect_threshold": practical_effect,
        "passes_effect": observed >= practical_effect,
        "passes_ci": bool(ci_low > 0),
        "passes_p": p_value < alpha,
        "ship": bool(observed >= practical_effect and ci_low > 0 and p_value < alpha),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted: NDArray[np.float64] = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, original_index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[original_index])
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted.tolist()
