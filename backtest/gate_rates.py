"""
backtest/gate_rates.py
----------------------
THE GATE for rate-source changes. Runs the per-batter prop backtest twice on the
SAME games with the SAME sim seeds -- once per rate source -- and paired-bootstraps
the difference. Nothing ships without p < 0.05.

  python -m backtest.gate_rates --a observed --b xrates --limit 60 --recent

Sources: observed | xrates | asof            (asof = point-in-time blend)

Metrics (lower is better, per batter-game):
  HR  : Brier score, (p_hr - homered)^2
  TB  : squared error, (exp_tb - actual_tb)^2

Resampling is at the GAME level, not the batter level: nine hitters in one game
share a park, a pitcher and a night, so their errors are correlated and treating
them as independent would fake significance.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation
from backtest.batter_props_backtest import fetch_batters

N_BOOT = 10000


def _tables(source: str, rate_season: int, date: str | None = None):
    if source == "observed":
        return load_rate_tables(rate_season)
    if source == "xrates":
        return load_rate_tables(rate_season, xrates=True)
    if source == "asof":
        from features.rates_asof import rates_asof
        return rates_asof(date, prior_season=rate_season)
    raise ValueError(f"unknown source {source}")


def collect(source: str, results, rate_season: int, n_sims: int):
    """Per-GAME lists of squared errors for HR (Brier) and TB."""
    per_game_hr, per_game_tb = [], []
    tables = None if source == "asof" else _tables(source, rate_season)
    cur_date = None
    for g in results:
        if source == "asof" and g["date"] != cur_date:
            cur_date = g["date"]
            tables = _tables("asof", rate_season, cur_date)
        fb = fetch_batters(g["gamePk"])
        if not fb:
            per_game_hr.append(None); per_game_tb.append(None); continue
        home = build_team(g["home"], fb["lineups"]["home"],
                          fb["starters"]["home"][0], fb["starters"]["home"][1], tables)
        away = build_team(g["away"], fb["lineups"]["away"],
                          fb["starters"]["away"][0], fb["starters"]["away"][1], tables)
        res = run_simulation(home, away, GameContext(park_code=g["home"]),
                             n_sims=n_sims, seed=g["gamePk"] % 9999)   # SAME seed both arms
        hr_err, tb_err = [], []
        for side, key in (("home", "home_batters"), ("away", "away_batters")):
            preds = res[key]
            for i, (pid, _) in enumerate(fb["lineups"][side]):
                if pid not in fb["actual"][side]:
                    continue
                a_hr, a_tb = fb["actual"][side][pid]
                hr_err.append((preds[i]["p_hr"] - a_hr) ** 2)
                tb_err.append((preds[i]["exp_tb"] - a_tb) ** 2)
        per_game_hr.append(float(np.mean(hr_err)) if hr_err else None)
        per_game_tb.append(float(np.mean(tb_err)) if tb_err else None)
    return per_game_hr, per_game_tb


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """Return (mean_delta B-A, win_rate_for_B, two-sided p) over game resamples."""
    n = len(a)
    diff = b - a                                  # negative => B better
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = diff[idx].mean(axis=1)
    mean_delta = float(diff.mean())
    win = float((boot < 0).mean())
    p = 2 * min(win, 1 - win)                     # two-sided
    return mean_delta, win, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="observed", help="baseline rate source")
    ap.add_argument("--b", default="xrates", help="challenger rate source")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--rate-season", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--recent", action="store_true")
    args = ap.parse_args()

    results = json.loads((config.SNAPSHOTS / f"results_{args.season}.json").read_text())
    results = [g for g in results if g.get("home_score") is not None]
    results = results[-args.limit:] if args.recent else results[:args.limit]
    print(f"gate: {args.b} (challenger) vs {args.a} (champion) over {len(results)} games\n")

    print(f"[1/2] running champion: {args.a}")
    a_hr, a_tb = collect(args.a, results, args.rate_season, args.sims)
    print(f"[2/2] running challenger: {args.b}")
    b_hr, b_tb = collect(args.b, results, args.rate_season, args.sims)

    rng = np.random.default_rng(12345)
    for label, A, B in (("HR (Brier)", a_hr, b_hr), ("TB (sq err)", a_tb, b_tb)):
        keep = [i for i in range(len(A)) if A[i] is not None and B[i] is not None]
        x = np.array([A[i] for i in keep]); y = np.array([B[i] for i in keep])
        d, win, p = paired_bootstrap(x, y, rng)
        verdict = ("SHIP" if p < 0.05 and d < 0 else
                   "REJECT (challenger worse)" if p < 0.05 else "HOLD (not significant)")
        print(f"\n{label} over {len(x)} games")
        print(f"  champion   {x.mean():.5f}")
        print(f"  challenger {y.mean():.5f}   delta {d:+.5f}")
        print(f"  challenger better in {win:.1%} of {N_BOOT} resamples, p={p:.3f}")
        print(f"  => {verdict}")


if __name__ == "__main__":
    main()