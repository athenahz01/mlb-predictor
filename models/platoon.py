"""Player-level handedness splits with league-effect partial pooling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import config

EVENTS = tuple(config.EVENTS)


@dataclass(frozen=True)
class SplitLine:
    pa: int
    rates: Mapping[str, float]


@dataclass(frozen=True)
class PlatoonProjection:
    vs_left: dict[str, float]
    vs_right: dict[str, float]
    effective_pa_left: int
    effective_pa_right: int
    model_version: str = "player-platoon-partial-pooling-v1"


def batting_side(batter_hand: str, pitcher_hand: str) -> str:
    """Resolve a switch hitter to the side used against this pitcher."""
    if batter_hand.upper() == "S":
        return "R" if pitcher_hand.upper() == "L" else "L"
    return batter_hand.upper()


def _normalize(rates: Mapping[str, float]) -> dict[str, float]:
    values = {
        event: max(0.0, float(rates.get(event, config.LEAGUE_PA_RATES[event]))) for event in EVENTS
    }
    total = sum(values.values())
    return {event: value / total for event, value in values.items()}


def shrink_split(
    observed: SplitLine | None,
    league_split: Mapping[str, float],
    *,
    prior_pa: float = 250.0,
) -> dict[str, float]:
    """Dirichlet posterior mean for one side of a player split."""
    league = _normalize(league_split)
    if observed is None or observed.pa <= 0:
        return league
    player = _normalize(observed.rates)
    denominator = observed.pa + prior_pa
    return _normalize(
        {
            event: (observed.pa * player[event] + prior_pa * league[event]) / denominator
            for event in EVENTS
        }
    )


def project_platoon(
    vs_left: SplitLine | None,
    vs_right: SplitLine | None,
    *,
    league_vs_left: Mapping[str, float],
    league_vs_right: Mapping[str, float],
    prior_pa: float = 250.0,
) -> PlatoonProjection:
    return PlatoonProjection(
        vs_left=shrink_split(vs_left, league_vs_left, prior_pa=prior_pa),
        vs_right=shrink_split(vs_right, league_vs_right, prior_pa=prior_pa),
        effective_pa_left=0 if vs_left is None else vs_left.pa,
        effective_pa_right=0 if vs_right is None else vs_right.pa,
    )
