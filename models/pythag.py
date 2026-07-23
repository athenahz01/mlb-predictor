"""
models/pythag.py
----------------
Pythagenpat baseline. Converts a team's runs scored/allowed into an expected
win%, then combines two teams' win% into a game probability via log5.

  exponent x = ((R + RA) / G) ** 0.287          (Pythagenpat; variable exponent)
  winpct     = R**x / (R**x + RA**x)
  log5 P(A beats B) = (a - a*b) / (a + b - 2*a*b)

This is a season-strength baseline: it knows nothing about today's starter, so
it's deliberately weak. Its job is to be a floor the simulation must clear.
"""

from __future__ import annotations


def pythagenpat_exponent(runs: float, runs_allowed: float, games: int) -> float:
    if games <= 0:
        return 2.0
    rpg = (runs + runs_allowed) / games
    return max(rpg, 0.01) ** 0.287


def win_pct(runs: float, runs_allowed: float, games: int) -> float:
    x = pythagenpat_exponent(runs, runs_allowed, games)
    rx, rax = runs**x, runs_allowed**x
    if rx + rax == 0:
        return 0.5
    return rx / (rx + rax)


def log5(a: float, b: float) -> float:
    """P(team with true win% a beats team with true win% b), neutral site."""
    denom = a + b - 2 * a * b
    if denom == 0:
        return 0.5
    return (a - a * b) / denom


def home_win_prob(home_wp: float, away_wp: float, hfa: float = 0.035) -> float:
    """log5 matchup with a small home-field bump (~0.035 win% ≈ 54% baseline)."""
    p = log5(home_wp, away_wp)
    return min(max(p + hfa, 0.01), 0.99)


def predict(home_stats: dict, away_stats: dict, hfa: float = 0.035) -> float:
    """
    home_stats / away_stats: {'R':runs, 'RA':runs_allowed, 'G':games}.
    Returns P(home win).
    """
    h = win_pct(home_stats["R"], home_stats["RA"], home_stats["G"])
    a = win_pct(away_stats["R"], away_stats["RA"], away_stats["G"])
    return home_win_prob(h, a, hfa)


if __name__ == "__main__":
    # strong home team vs weak away team
    home = {"R": 420, "RA": 350, "G": 80}
    away = {"R": 330, "RA": 430, "G": 80}
    print("home win% :", round(win_pct(home["R"], home["RA"], home["G"]), 3))
    print("away win% :", round(win_pct(away["R"], away["RA"], away["G"]), 3))
    print("P(home)   :", round(predict(home, away), 3))
