"""
features/pa_probabilities.py
----------------------------
Turn a (batter, pitcher) matchup into a per-plate-appearance probability vector
over the 8 canonical outcomes, then apply park / platoon / umpire adjustments.

Method: the multinomial generalisation of Bill James' log5, a.k.a. Tango's
"odds ratio" method. For each outcome o:

    OR_o = odds(batter_o) * odds(pitcher_o) / odds(league_o)
    rate_o = OR_o / (1 + OR_o)

then renormalise the 8 rates to sum to 1.

Documented caveat (Morey & Cohen, J. Sports Analytics 2015): log5 is skewed for
extreme/asymmetric rates and OVER-estimates HR% for outlier bats/arms. We apply
a shrink toward league on the HR term for extreme matchups. Validate prop tails
against realised frequencies before trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import config
from models.platoon import batting_side

if TYPE_CHECKING:
    from models.batter_playing_time import BatterPlayingTime

EVENTS = config.EVENTS
L = config.LEAGUE_PA_RATES


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


@dataclass
class Batter:
    name: str
    rates: dict[str, float]  # per-PA rates over EVENTS (need not sum to 1)
    hand: str = "R"  # R / L / S(switch)
    order: int = 0
    mlb_id: int | None = None
    data_quality_flags: tuple[str, ...] = field(default_factory=tuple)
    platoon_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    playing_time: BatterPlayingTime | None = None
    pitch_type_factors: dict[int, float] = field(default_factory=dict)

    def vector(self, pitcher_hand: str | None = None) -> dict[str, float]:
        if pitcher_hand:
            split = self.platoon_rates.get(f"vs_{pitcher_hand.upper()}")
            if split:
                return {e: split.get(e, self.rates.get(e, L[e])) for e in EVENTS}
        return {e: self.rates.get(e, L[e]) for e in EVENTS}


@dataclass
class Pitcher:
    name: str
    rates: dict[str, float]  # per-PA rates allowed over EVENTS
    hand: str = "R"
    is_starter: bool = True
    mlb_id: int | None = None
    data_quality_flags: tuple[str, ...] = field(default_factory=tuple)
    platoon_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    role: str | None = None

    def vector(self, batter_side: str | None = None) -> dict[str, float]:
        if batter_side:
            split = self.platoon_rates.get(f"vs_{batter_side.upper()}")
            if split:
                return {e: split.get(e, self.rates.get(e, L[e])) for e in EVENTS}
        return {e: self.rates.get(e, L[e]) for e in EVENTS}


_MEASURED_PLATOON = None


def _load_measured_platoon():
    global _MEASURED_PLATOON
    if _MEASURED_PLATOON is None:
        try:
            import json

            _MEASURED_PLATOON = json.loads((config.SNAPSHOTS / "platoon_mults.json").read_text())
        except Exception:
            _MEASURED_PLATOON = {}
    return _MEASURED_PLATOON


def _platoon_mult(bat_hand: str, pit_hand: str) -> dict[str, float]:
    """
    Platoon multipliers. If data/snapshots/platoon_mults.json exists (built by
    ingest/build_platoon.py from real league splits), use the measured values;
    otherwise fall back to conservative directional priors.
    """
    if bat_hand == "S":
        advantage = True
    else:
        advantage = bat_hand != pit_hand
    measured = _load_measured_platoon()
    key = "advantage" if advantage else "same_hand"
    if measured.get(key):
        return measured[key]
    if advantage:
        return {
            "BB": 1.05,
            "1B": 1.03,
            "2B": 1.04,
            "3B": 1.04,
            "HR": 1.08,
            "HBP": 1.0,
            "K": 0.95,
            "IP_OUT": 1.0,
        }
    return {
        "BB": 0.96,
        "1B": 0.98,
        "2B": 0.97,
        "3B": 0.97,
        "HR": 0.93,
        "HBP": 1.0,
        "K": 1.06,
        "IP_OUT": 1.0,
    }


def matchup_rates(
    batter: Batter,
    pitcher: Pitcher,
    park_code: str = "_DEFAULT",
    ump_k_mult: float = 1.0,
    tto_bump: float = 0.0,
    hr_shrink: float = 0.15,
    env_hr: float = 1.0,
    env_hit: float = 1.0,
) -> dict[str, float]:
    """
    Return a normalised per-PA probability dict over EVENTS for this matchup.

    park_code  : key into config.PARK_FACTORS (HR + hit multipliers).
    ump_k_mult : multiply K rate by this (umpire strike-zone tendency).
    tto_bump   : additive offense nudge applied each time through the order (3rd+).
    hr_shrink  : 0..1 pull of the combined HR odds toward league (Morey-Cohen fix).
    env_hr/env_hit : game-day environment multipliers (weather; 1.0 = neutral).
    """
    resolved_batter_side = batting_side(batter.hand, pitcher.hand)
    b = batter.vector(pitcher.hand)
    p = pitcher.vector(resolved_batter_side)
    uses_player_split = bool(
        batter.platoon_rates.get(f"vs_{pitcher.hand.upper()}")
        or pitcher.platoon_rates.get(f"vs_{resolved_batter_side}")
    )

    # 1) multinomial log5 via odds ratios
    raw = {}
    for e in EVENTS:
        orr = _odds(b[e]) * _odds(p[e]) / _odds(L[e])
        if e == "HR" and hr_shrink > 0:
            # shrink the *odds* toward the league odds for the HR term only
            orr = orr ** (1 - hr_shrink) * _odds(L["HR"]) ** hr_shrink
        raw[e] = orr / (1 + orr)

    # 2) platoon
    if not uses_player_split:
        plat = _platoon_mult(batter.hand, pitcher.hand)
        for e in EVENTS:
            raw[e] *= plat[e]

    pitch_type_factor = (
        batter.pitch_type_factors.get(pitcher.mlb_id, 1.0) if pitcher.mlb_id is not None else 1.0
    )
    if pitch_type_factor != 1.0:
        for event in ("1B", "2B", "3B", "HR"):
            raw[event] *= pitch_type_factor
        raw["K"] *= max(0.90, min(1.10, 2.0 - pitch_type_factor))

    # 3) park (HR factor on HR; hit factor on balls that fall for hits)
    #    + game-day environment (weather) multipliers on the same axes
    pf = config.park(park_code)
    raw["HR"] *= pf["hr"] * env_hr
    for e in ("1B", "2B", "3B"):
        raw[e] *= pf["hit"] * env_hit

    # 4) umpire K tendency
    raw["K"] *= ump_k_mult

    # 5) times-through-order: small uniform offense bump (and matching K dip)
    if tto_bump:
        for e in ("BB", "1B", "2B", "HR"):
            raw[e] *= 1 + tto_bump
        raw["K"] *= 1 - tto_bump

    # 6) renormalise to a proper distribution
    tot = sum(raw.values())
    return {e: raw[e] / tot for e in EVENTS}


# --------------------------------------------------------------------------
# Convenience: build rate dicts from headline stats so you can prototype
# without a full Statcast pull.
# --------------------------------------------------------------------------
def batter_from_slash(
    name, *, k_pct, bb_pct, hr_pct, hand="R", x1b=None, x2b=None, x3b=None, order=0
) -> Batter:
    """Build a Batter from K%, BB%, HR% (as decimals); the rest fills from league
    shape scaled to the remaining probability mass."""
    rest = 1 - k_pct - bb_pct - hr_pct - L["HBP"]
    league_hit_shape = {e: L[e] for e in ("1B", "2B", "3B", "IP_OUT")}
    shape_tot = sum(league_hit_shape.values())
    rates = {"K": k_pct, "BB": bb_pct, "HR": hr_pct, "HBP": L["HBP"]}
    for e in ("1B", "2B", "3B", "IP_OUT"):
        rates[e] = rest * league_hit_shape[e] / shape_tot
    if x1b:
        rates["1B"] = x1b
    if x2b:
        rates["2B"] = x2b
    if x3b:
        rates["3B"] = x3b
    return Batter(name=name, rates=rates, hand=hand, order=order)


def pitcher_from_rates(name, *, k_pct, bb_pct, hr_pct, hand="R", is_starter=True) -> Pitcher:
    rest = 1 - k_pct - bb_pct - hr_pct - L["HBP"]
    league_hit_shape = {e: L[e] for e in ("1B", "2B", "3B", "IP_OUT")}
    shape_tot = sum(league_hit_shape.values())
    rates = {"K": k_pct, "BB": bb_pct, "HR": hr_pct, "HBP": L["HBP"]}
    for e in ("1B", "2B", "3B", "IP_OUT"):
        rates[e] = rest * league_hit_shape[e] / shape_tot
    return Pitcher(name=name, rates=rates, hand=hand, is_starter=is_starter)
