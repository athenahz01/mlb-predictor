"""
backtest/diagnose.py
--------------------
Instrumented single-pass over a few completed games to find the backtest bug.
For each game it prints the matchup, both starters, how many batters actually
joined the 2025 rate table (vs fell back to league average), the sim's P(home),
and who actually won. Bypasses the cache.

  python -m backtest.diagnose --season 2026 --rate-season 2025 --limit 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np

import config
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation
from backtest.sim_backtest import fetch_lineups_starters


def join_count(lineup, table):
    return sum(1 for pid, _ in lineup if pid in table.index)


def run(season, rate_season, limit):
    games = json.loads((config.SNAPSHOTS / f"results_{season}.json").read_text())
    games.sort(key=lambda g: (g["date"], g["gamePk"]))
    tables = load_rate_tables(rate_season)
    print(f"rate table: {len(tables['bat'])} batters, {len(tables['pit'])} pitchers")
    print(f"index dtype: {tables['bat'].index.dtype}\n")

    sims, ys = [], []
    for g in games[:limit]:
        lu = fetch_lineups_starters(g["gamePk"])
        if not lu:
            print(f"{g['away']}@{g['home']}: no lineups"); continue
        (h_line, h_sp), (a_line, a_sp) = lu["home"], lu["away"]
        h_join = join_count(h_line, tables["bat"])
        a_join = join_count(a_line, tables["bat"])
        h_sp_join = h_sp[0] in tables["pit"].index
        a_sp_join = a_sp[0] in tables["pit"].index
        home = build_team(g["home"], h_line, h_sp[0], h_sp[1], tables)
        away = build_team(g["away"], a_line, a_sp[0], a_sp[1], tables)
        ctx = GameContext(park_code=g["home"])
        res = run_simulation(home, away, ctx, n_sims=800, seed=g["gamePk"] % 9999)
        sim_p = res["p_home_win"]
        y = int(g["home_score"] > g["away_score"])
        sims.append(sim_p); ys.append(y)
        print(f"{g['away']:>4}@{g['home']:<4} "
              f"score {g['away_score']}-{g['home_score']} "
              f"won={'HOME' if y else 'AWAY'} | "
              f"sim P(home)={sim_p:.3f} | "
              f"bat joins H{h_join}/9 A{a_join}/9 | "
              f"SP join H{'Y' if h_sp_join else 'n'} A{'Y' if a_sp_join else 'n'} | "
              f"SP {h_sp[1]} vs {a_sp[1]}")

    if len(sims) > 2:
        corr = np.corrcoef(sims, ys)[0, 1]
        print(f"\ncorr(sim_p, home_win) = {corr:+.3f}  "
              f"({'INVERTED' if corr < 0 else 'aligned'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--rate-season", type=int, default=dt.date.today().year - 1)
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    run(args.season, args.rate_season, args.limit)
