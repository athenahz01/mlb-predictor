"""Leakage-safe hierarchical player event-rate projections.

The projection is intentionally a pure function.  Callers must pass only season
lines that were knowable at the prediction timestamp.  This makes the same code
usable by live prediction and chronological walk-forward evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from statistics import NormalDist

import config

EVENTS = tuple(config.EVENTS)

# Event rates stabilize at very different speeds.  These are preregistered
# challenger hyperparameters, not production claims; evaluation decides whether
# a category may use them.
EVENT_STABILIZATION_PA: dict[str, float] = {
    "BB": 120.0,
    "HBP": 600.0,
    "1B": 370.0,
    "2B": 550.0,
    "3B": 1_500.0,
    "HR": 170.0,
    "K": 60.0,
    "IP_OUT": 220.0,
}

RECENCY_WEIGHTS: dict[int, float] = {0: 1.0, 1: 0.80, 2: 0.55, 3: 0.35}
CONTACT_EVENTS = frozenset({"1B", "2B", "3B", "HR", "IP_OUT"})


@dataclass(frozen=True)
class SeasonLine:
    season: int
    pa: int
    rates: Mapping[str, float]
    expected_rates: Mapping[str, float] | None = None
    injury_return: bool = False


@dataclass(frozen=True)
class EventPosterior:
    mean: float
    lower_95: float
    upper_95: float
    effective_pa: float
    prior_pa: float


@dataclass(frozen=True)
class PlayerProjection:
    rates: dict[str, float]
    uncertainty: dict[str, EventPosterior]
    effective_pa: float
    model_version: str = "hierarchical-marcel-v1"
    data_quality_flags: tuple[str, ...] = field(default_factory=tuple)


def _normalized(rates: Mapping[str, float]) -> dict[str, float]:
    values = {event: max(0.0, float(rates.get(event, 0.0))) for event in EVENTS}
    total = sum(values.values())
    if total <= 0:
        return dict(config.LEAGUE_PA_RATES)
    return {event: value / total for event, value in values.items()}


def _age_multiplier(event: str, age: float | None) -> float:
    """Conservative Marcel-style aging centered on a typical age-29 season."""
    if age is None:
        return 1.0
    delta = age - 29.0
    if event in {"1B", "2B", "3B"}:
        annual = -0.004 if delta > 0 else 0.002
    elif event == "HR":
        annual = -0.003 if delta > 0 else 0.003
    elif event == "K":
        annual = 0.002 if delta > 0 else -0.001
    elif event == "BB":
        annual = 0.001 if delta > 0 else -0.001
    else:
        annual = 0.0
    return max(0.90, min(1.10, 1.0 + abs(delta) * annual))


def project_player(
    lines: Sequence[SeasonLine],
    *,
    target_season: int,
    age: float | None = None,
    league_rates: Mapping[str, float] | None = None,
    expected_contact_weight: float = 0.35,
) -> PlayerProjection:
    """Return event-specific beta-binomial posterior means and uncertainty.

    The eight marginal posteriors are normalized at the end to preserve a valid
    multinomial plate-appearance distribution. Expected contact quality affects
    only contact events and is capped so observed outcomes remain dominant.
    """
    if not 0.0 <= expected_contact_weight <= 0.5:
        raise ValueError("expected_contact_weight must be between 0 and 0.5")
    league = _normalized(league_rates or config.LEAGUE_PA_RATES)
    flags: set[str] = set()
    if not lines:
        flags.add("missing_player_history_league_prior")
    if age is None:
        flags.add("missing_age_no_adjustment")

    raw_means: dict[str, float] = {}
    raw_posteriors: dict[str, EventPosterior] = {}
    total_effective_pa = 0.0

    for event in EVENTS:
        successes = 0.0
        trials = 0.0
        for line in lines:
            if line.pa <= 0 or line.season > target_season:
                continue
            seasons_ago = target_season - line.season
            weight = RECENCY_WEIGHTS.get(seasons_ago, 0.0)
            if weight == 0.0:
                continue
            observed = _normalized(line.rates)[event]
            rate = observed
            if event in CONTACT_EVENTS and line.expected_rates:
                expected = _normalized(line.expected_rates)[event]
                rate = (
                    1.0 - expected_contact_weight
                ) * observed + expected_contact_weight * expected
            # Return-from-injury records stay useful but receive less exposure.
            exposure_mult = 0.70 if line.injury_return else 1.0
            effective = float(line.pa) * weight * exposure_mult
            successes += effective * rate
            trials += effective

        prior_pa = EVENT_STABILIZATION_PA[event]
        alpha = prior_pa * league[event] + successes
        beta = prior_pa * (1.0 - league[event]) + trials - successes
        mean = alpha / (alpha + beta)
        mean *= _age_multiplier(event, age)
        raw_means[event] = mean

        posterior_n = alpha + beta
        variance = alpha * beta / (posterior_n**2 * (posterior_n + 1.0))
        radius = NormalDist().inv_cdf(0.975) * sqrt(max(variance, 0.0))
        raw_posteriors[event] = EventPosterior(
            mean=mean,
            lower_95=max(0.0, mean - radius),
            upper_95=min(1.0, mean + radius),
            effective_pa=trials,
            prior_pa=prior_pa,
        )
        total_effective_pa = max(total_effective_pa, trials)

    rates = _normalized(raw_means)
    uncertainty = {
        event: EventPosterior(
            mean=rates[event],
            lower_95=raw_posteriors[event].lower_95,
            upper_95=raw_posteriors[event].upper_95,
            effective_pa=raw_posteriors[event].effective_pa,
            prior_pa=raw_posteriors[event].prior_pa,
        )
        for event in EVENTS
    }
    if any(line.expected_rates is None for line in lines):
        flags.add("partial_expected_contact_quality")
    if any(line.injury_return for line in lines):
        flags.add("injury_return_exposure_discount")
    return PlayerProjection(
        rates=rates,
        uncertainty=uncertainty,
        effective_pa=total_effective_pa,
        data_quality_flags=tuple(sorted(flags)),
    )
