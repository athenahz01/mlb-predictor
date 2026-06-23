"""
models/features.py
------------------
Build a leakage-free training table from season results. We walk games in
chronological order; for each game we compute features from ONLY the games that
happened before it (running Elo + running Pythagenpat), record the row, and only
THEN update the running state with the game's outcome.

This is the dataset the Stage 2 ensemble trains on, and it also stores each
game's Elo win probability so we can significance-test the ensemble against the
Elo baseline on the same games.

Columns out:
  elo_diff       home_elo - away_elo (pre-game)
  pythag_diff    home_pythag_winpct - away_pythag_winpct (pre-game, season-to-date)
  home_rest      days since home team's last game (capped)
  away_rest      days since away team's last game (capped)
  elo_p          Elo's pre-game P(home win)  -> baseline column, not a feature
  home_win       target (1/0)
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

import config
from models.elo import EloModel
from models import pythag

FEATURE_COLS = ["elo_diff", "pythag_diff", "home_rest", "away_rest"]


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def build_training_table(season: int, burn_in: int = 100) -> pd.DataFrame:
    """
    season: reads data/snapshots/results_<season>.json (from pull_results).
    burn_in: drop the first N games (Elo/Pythag still stabilizing) from training.
    """
    rpath = config.SNAPSHOTS / f"results_{season}.json"
    if not rpath.exists():
        raise FileNotFoundError(
            f"No results snapshot. Run: python -m ingest.pull_results --season {season}")
    games = json.loads(rpath.read_text())
    games.sort(key=lambda g: (g["date"], g["gamePk"]))

    elo = EloModel()
    runs = {}                       # team -> {R, RA, G}
    last_game = {}                  # team -> date
    rows = []

    for i, g in enumerate(games):
        h, a = g["home"], g["away"]
        d = _date(g["date"])

        # --- features from state BEFORE this game ---
        elo_diff = elo.rating(h) - elo.rating(a)
        hp = runs.get(h); ap = runs.get(a)
        if hp and ap and hp["G"] > 0 and ap["G"] > 0:
            pyth_h = pythag.win_pct(hp["R"], hp["RA"], hp["G"])
            pyth_a = pythag.win_pct(ap["R"], ap["RA"], ap["G"])
            pythag_diff = pyth_h - pyth_a
        else:
            pythag_diff = 0.0
        home_rest = min((d - last_game[h]).days, 7) if h in last_game else 3
        away_rest = min((d - last_game[a]).days, 7) if a in last_game else 3
        elo_p = elo.predict(h, a)

        rows.append({
            "date": g["date"], "home": h, "away": a,
            "elo_diff": elo_diff, "pythag_diff": pythag_diff,
            "home_rest": home_rest, "away_rest": away_rest,
            "elo_p": elo_p,
            "home_win": int(g["home_score"] > g["away_score"]),
            "idx": i,
        })

        # --- now update state with the outcome ---
        elo.update(h, a, g["home_score"], g["away_score"])
        for team, rf, ra in ((h, g["home_score"], g["away_score"]),
                             (a, g["away_score"], g["home_score"])):
            r = runs.setdefault(team, {"R": 0, "RA": 0, "G": 0})
            r["R"] += rf; r["RA"] += ra; r["G"] += 1
        last_game[h] = d; last_game[a] = d

    df = pd.DataFrame(rows)
    if burn_in:
        df = df[df["idx"] >= burn_in].reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    df = build_training_table(args.season)
    print(df[["date", "home", "away", *FEATURE_COLS, "elo_p", "home_win"]].head(10).to_string())
    print(f"\n{len(df)} training rows after burn-in")
