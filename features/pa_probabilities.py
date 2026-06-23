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
from typing import Dict

import numpy as np

import config

EVENTS = config.EVENTS
L = config.LEAGUE_PA_RATES


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


@dataclass
class Batter:
    name: str
    rates: Dict[str, float]            # per-PA rates over EVENTS (need not sum to 1)
    hand: str = "R"                    # R / L / S(switch)
    order: int = 0

    def vector(self) -> Dict[str, float]:
        return {e: self.rates.get(e, L[e]) for e in EVENTS}


@dataclass
class Pitcher:
    name: str
    rates: Dict[str, float]            # per-PA rates allowed over EVENTS
    hand: str = "R"
    is_starter: bool = True

    def vector(self) -> Dict[str, float]:
        return {e: self.rates.get(e, L[e]) for e in EVENTS}


def _platoon_mult(bat_hand: str, pit_hand: str) -> Dict[str, float]:
    """
    Crude, directional platoon multipliers. A batter facing the OPPOSITE hand
    (the platoon advantage) gets a small offense bump; same-hand gets a haircut.
    Switch hitters always take the advantage. Magnitudes are conservative; refine
    with real L/R splits per batter when you have them.
    """
    if bat_hand == "S":
        advantage = True
    else:
        advantage = (bat_hand != pit_hand)
    if advantage:
        return {"BB": 1.05, "1B": 1.03, "2B": 1.04, "3B": 1.04, "HR": 1.08,
                "HBP": 1.0, "K": 0.95, "IP_OUT": 1.0}
    return {"BB": 0.96, "1B": 0.98, "2B": 0.97, "3B": 0.97, "HR": 0.93,
            "HBP": 1.0, "K": 1.06, "IP_OUT": 1.0}


def matchup_rates(
    batter: Batter,
    pitcher: Pitcher,
    park_code: str = "_DEFAULT",
    ump_k_mult: float = 1.0,
    tto_bump: float = 0.0,
    hr_shrink: float = 0.15,
) -> Dict[str, float]:
    """
    Return a normalised per-PA probability dict over EVENTS for this matchup.

    park_code  : key into config.PARK_FACTORS (HR + hit multipliers).
    ump_k_mult : multiply K rate by this (umpire strike-zone tendency).
    tto_bump   : additive offense nudge applied each time through the order (3rd+).
    hr_shrink  : 0..1 pull of the combined HR odds toward league (Morey-Cohen fix).
    """
    b = batter.vector()
    p = pitcher.vector()

    # 1) multinomial log5 via odds ratios
    raw = {}
    for e in EVENTS:
        orr = _odds(b[e]) * _odds(p[e]) / _odds(L[e])
        if e == "HR" and hr_shrink > 0:
            # shrink the *odds* toward the league odds for the HR term only
            orr = orr ** (1 - hr_shrink) * _odds(L["HR"]) ** hr_shrink
        raw[e] = orr / (1 + orr)

    # 2) platoon
    plat = _platoon_mult(batter.hand, pitcher.hand)
    for e in EVENTS:
        raw[e] *= plat[e]

    # 3) park (HR factor on HR; hit factor on balls that fall for hits)
    pf = config.park(park_code)
    raw["HR"] *= pf["hr"]
    for e in ("1B", "2B", "3B"):
        raw[e] *= pf["hit"]

    # 4) umpire K tendency
    raw["K"] *= ump_k_mult

    # 5) times-through-order: small uniform offense bump (and matching K dip)
    if tto_bump:
        for e in ("BB", "1B", "2B", "HR"):
            raw[e] *= (1 + tto_bump)
        raw["K"] *= (1 - tto_bump)

    # 6) renormalise to a proper distribution
    tot = sum(raw.values())
    return {e: raw[e] / tot for e in EVENTS}


# --------------------------------------------------------------------------
# Convenience: build rate dicts from headline stats so you can prototype
# without a full Statcast pull.
# --------------------------------------------------------------------------
def batter_from_slash(name, *, k_pct, bb_pct, hr_pct, hand="R",
                      x1b=None, x2b=None, x3b=None, order=0) -> Batter:
    """Build a Batter from K%, BB%, HR% (as decimals); the rest fills from league
    shape scaled to the remaining probability mass."""
    rest = 1 - k_pct - bb_pct - hr_pct - L["HBP"]
    league_hit_shape = {e: L[e] for e in ("1B", "2B", "3B", "IP_OUT")}
    shape_tot = sum(league_hit_shape.values())
    rates = {"K": k_pct, "BB": bb_pct, "HR": hr_pct, "HBP": L["HBP"]}
    for e in ("1B", "2B", "3B", "IP_OUT"):
        rates[e] = rest * league_hit_shape[e] / shape_tot
    if x1b: rates["1B"] = x1b
    if x2b: rates["2B"] = x2b
    if x3b: rates["3B"] = x3b
    return Batter(name=name, rates=rates, hand=hand, order=order)


def pitcher_from_rates(name, *, k_pct, bb_pct, hr_pct, hand="R",
                       is_starter=True) -> Pitcher:
    rest = 1 - k_pct - bb_pct - hr_pct - L["HBP"]
    league_hit_shape = {e: L[e] for e in ("1B", "2B", "3B", "IP_OUT")}
    shape_tot = sum(league_hit_shape.values())
    rates = {"K": k_pct, "BB": bb_pct, "HR": hr_pct, "HBP": L["HBP"]}
    for e in ("1B", "2B", "3B", "IP_OUT"):
        rates[e] = rest * league_hit_shape[e] / shape_tot
    return Pitcher(name=name, rates=rates, hand=hand, is_starter=is_starter)
