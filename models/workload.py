"""Starting-pitcher workload distributions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import erfc, sqrt
from statistics import pstdev

import numpy as np


@dataclass(frozen=True)
class StarterWorkload:
    expected_pitches: float
    pitch_sd: float
    minimum_pitches: int = 35
    maximum_pitches: int = 115
    pitches_per_batter: float = 3.85
    opener_probability: float = 0.0
    data_quality_flags: tuple[str, ...] = field(default_factory=tuple)
    model_version: str = "starter-workload-v1"

    def sample_pitch_limit(self, rng: np.random.Generator) -> int:
        if rng.random() < self.opener_probability:
            return int(rng.integers(20, 46))
        draw = rng.normal(self.expected_pitches, max(self.pitch_sd, 1.0))
        return int(round(np.clip(draw, self.minimum_pitches, self.maximum_pitches)))

    @property
    def expected_batters_faced(self) -> float:
        return self.expected_pitches / self.pitches_per_batter

    @property
    def expected_innings(self) -> float:
        return self.expected_batters_faced * 0.72 / 3.0

    def probability_starting_inning(self, inning: int) -> float:
        """Normal approximation to surviving through the prior inning."""
        if inning <= 1:
            return 1.0
        threshold = (inning - 1) * 3.0 / 0.72 * self.pitches_per_batter
        if self.pitch_sd <= 0:
            return float(self.expected_pitches >= threshold)
        z = (threshold - self.expected_pitches) / self.pitch_sd
        # Stable normal survival approximation without scipy.
        return float(0.5 * erfc(z / sqrt(2.0)))


def fit_starter_workload(
    recent_pitch_counts: Sequence[int],
    *,
    days_rest: int | None = None,
    season_high: int | None = None,
    injury_return: bool = False,
    role: str = "starter",
    manager_quick_hook: float = 0.0,
) -> StarterWorkload:
    """Fit a truncated workload distribution from pregame-available context."""
    valid = [int(value) for value in recent_pitch_counts if 0 < int(value) <= 140]
    flags: set[str] = set()
    if valid:
        recent = valid[-5:]
        weights: np.ndarray = np.arange(1, len(recent) + 1, dtype=float)
        expected = float(np.average(recent, weights=weights))
        spread = max(6.0, pstdev(recent) if len(recent) > 1 else 9.0)
    else:
        expected = 90.0
        spread = 12.0
        flags.add("missing_recent_pitch_counts_league_prior")

    if days_rest is not None and days_rest < 4:
        expected -= 7.0 * (4 - days_rest)
        flags.add("short_rest_adjustment")
    if injury_return:
        expected = min(expected, 75.0)
        spread = max(spread, 12.0)
        flags.add("injury_return_workload_cap")
    expected -= max(-1.0, min(1.0, manager_quick_hook)) * 5.0

    opener_probability = 0.0
    if role.lower() == "opener":
        opener_probability = 1.0
        expected = min(expected, 35.0)
        spread = min(spread, 8.0)
        flags.add("opener_role")
    elif role.lower() == "bulk":
        expected = min(expected, 75.0)
        flags.add("bulk_role")

    maximum = min(120, season_high + 5) if season_high else 115
    expected = max(30.0, min(float(maximum), expected))
    return StarterWorkload(
        expected_pitches=expected,
        pitch_sd=spread,
        minimum_pitches=20 if role.lower() == "opener" else 35,
        maximum_pitches=maximum,
        opener_probability=opener_probability,
        data_quality_flags=tuple(sorted(flags)),
    )
