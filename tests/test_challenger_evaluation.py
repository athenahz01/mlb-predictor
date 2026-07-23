from __future__ import annotations

import pytest

from evaluation.challenger import count_crps, gate_challenger


def test_count_crps_rewards_distribution_near_actual():
    close = count_crps({"2": 0.2, "3": 0.6, "4": 0.2}, 3)
    far = count_crps({"8": 1.0}, 3)
    assert close < far
    assert count_crps({"3": 1.0}, 3) == pytest.approx(0.0)


def test_challenger_gate_requires_stability_and_practical_effect():
    dates = [f"2026-04-{day:02d}" for day in range(1, 11)]
    champion = [0.8] * 10
    challenger = [0.6] * 10
    result = gate_challenger(
        champion,
        challenger,
        dates,
        practical_effect=0.01,
        seed=4,
        n_boot=500,
    )
    assert result["stable_across_time_halves"]
    assert result["ship"]
