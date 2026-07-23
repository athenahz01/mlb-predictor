from __future__ import annotations

import numpy as np

from evaluation.metrics import (
    binary_metrics,
    date_clustered_paired_bootstrap,
    holm_adjust,
)


def test_calibration_metrics_for_perfect_predictions():
    metrics = binary_metrics([0.01, 0.99, 0.02, 0.98], [0, 1, 0, 1])
    assert metrics["brier"] < 0.001
    assert metrics["log_loss"] < 0.03


def test_date_clustered_bootstrap_is_seeded_and_requires_practical_effect():
    champion = np.full(20, 0.4)
    challenger = np.full(20, 0.39)
    dates = np.repeat(["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"], 5)
    first = date_clustered_paired_bootstrap(
        challenger, champion, dates, n_boot=500, seed=7, practical_effect=0.02
    )
    second = date_clustered_paired_bootstrap(
        challenger, champion, dates, n_boot=500, seed=7, practical_effect=0.02
    )
    assert first == second
    assert first["passes_ci"]
    assert not first["passes_effect"]
    assert not first["ship"]


def test_holm_adjustment_is_monotonic():
    assert holm_adjust([0.01, 0.04, 0.2]) == [0.03, 0.08, 0.2]
