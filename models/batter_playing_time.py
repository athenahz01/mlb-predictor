"""Starter batter plate-appearance survival distributions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BatterPlayingTime:
    """Probability the named starter owns each PA taken by the batting-order slot."""

    survival_by_slot_pa: tuple[float, ...]
    sample_games: int
    model_version: str = "starter-pa-survival-v1"

    def probability_active(self, slot_pa_number: int) -> float:
        if slot_pa_number <= 0:
            raise ValueError("slot_pa_number must be positive")
        index = slot_pa_number - 1
        if index >= len(self.survival_by_slot_pa):
            return 0.0
        return self.survival_by_slot_pa[index]


def fit_playing_time(
    starter_plate_appearances: Sequence[int],
    *,
    max_slot_pa: int = 8,
) -> BatterPlayingTime:
    """Fit an empirical survival curve from historical starts.

    Input games must be filtered to dates before the prediction. The fitted
    survival function is monotone by construction.
    """
    counts = [max(0, int(value)) for value in starter_plate_appearances]
    if not counts:
        # Explicit conservative prior, used only when the loader marks missing history.
        return BatterPlayingTime((1.0, 1.0, 0.98, 0.82, 0.28, 0.05, 0.01, 0.0), 0)
    n = len(counts)
    survival = tuple(
        sum(value >= pa_number for value in counts) / n for pa_number in range(1, max_slot_pa + 1)
    )
    return BatterPlayingTime(survival, n)
