"""
market/devig.py
---------------
Convert market prices (Kalshi cents or American odds) into FAIR (vig-free)
probabilities so model-vs-market edge is apples-to-apples.

Two methods:
  multiplicative : divide by the overround. Simple; biases toward favourites.
  power          : solve for k s.t. sum(p_i^k) = 1. More accurate on lopsided
                   MLB moneylines (removes more vig from the favourite).

For binary Kalshi markets you usually have yes/no implied probs that sum to
>1 (the overround); de-vig collapses them to a fair pair.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import brentq


def american_to_prob(odds: float) -> float:
    """American odds -> implied probability (with vig)."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def cents_to_prob(cents: float) -> float:
    """Kalshi price in cents (1..99) -> implied probability."""
    return cents / 100.0


def devig_multiplicative(probs: List[float]) -> List[float]:
    s = sum(probs)
    return [p / s for p in probs]


def devig_power(probs: List[float]) -> List[float]:
    """
    Power de-vig: find k with sum(p_i**k) = 1, return p_i**k.
    Favourites lose more vig than the multiplicative method assigns them.
    """
    probs = np.asarray(probs, float)
    if abs(probs.sum() - 1.0) < 1e-9:
        return list(probs)

    def f(k):
        return np.sum(probs ** k) - 1.0

    # overround>1 -> k>1 shrinks; underround -> k<1. Bracket generously.
    lo, hi = 0.5, 5.0
    try:
        k = brentq(f, lo, hi)
    except ValueError:
        return devig_multiplicative(list(probs))
    return list(probs ** k)


def fair_two_way(yes_price_cents: float, no_price_cents: float,
                 method: str = "power") -> dict:
    """
    Given Kalshi YES and NO ask prices (cents), return fair probabilities.
    Pass yes_ask and no_ask (what you'd pay to take each side).
    """
    p_yes = cents_to_prob(yes_price_cents)
    p_no = cents_to_prob(no_price_cents)
    fair = devig_power([p_yes, p_no]) if method == "power" \
        else devig_multiplicative([p_yes, p_no])
    return {"fair_yes": float(fair[0]), "fair_no": float(fair[1]),
            "overround": float(p_yes + p_no), "method": method}


def edge(model_p: float, fair_p: float) -> float:
    """Model edge in probability points (positive = model higher than market)."""
    return model_p - fair_p
