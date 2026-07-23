"""
models/baselines.py
-------------------
Tie the baselines together: load season results, build Elo + Pythagenpat, and
predict P(home win) for a matchup. These are the FLOOR the simulation must beat
at p<0.05 before any version ships.

  python -m models.baselines --game PIT@COL --season 2026

Requires a results snapshot first:
  python -m ingest.pull_results --season 2026
"""

from __future__ import annotations

import argparse
import json

import config
from models import pythag
from models.elo import EloModel


def load_results(season: int):
    rpath = config.SNAPSHOTS / f"results_{season}.json"
    apath = config.SNAPSHOTS / f"team_runs_{season}.json"
    if not rpath.exists():
        raise FileNotFoundError(
            f"No results snapshot. Run:  python -m ingest.pull_results --season {season}"
        )
    games = json.loads(rpath.read_text())
    agg = json.loads(apath.read_text()) if apath.exists() else {}
    return games, agg


def build_elo(games: list[dict]) -> EloModel:
    m = EloModel()
    m.run_games(games)
    return m


def predict(home: str, away: str, season: int) -> dict:
    games, agg = load_results(season)
    elo = build_elo(games)
    out = {"elo_home_p": round(elo.predict(home, away), 4)}
    if home in agg and away in agg:
        out["pythag_home_p"] = round(pythag.predict(agg[home], agg[away]), 4)
        out["home_winpct"] = round(
            pythag.win_pct(
                **{"runs": agg[home]["R"], "runs_allowed": agg[home]["RA"], "games": agg[home]["G"]}
            ),
            3,
        )
        out["away_winpct"] = round(
            pythag.win_pct(
                **{"runs": agg[away]["R"], "runs_allowed": agg[away]["RA"], "games": agg[away]["G"]}
            ),
            3,
        )
    else:
        out["pythag_home_p"] = None
        out["note"] = "team not in run aggregates yet"
    out["elo_home_rating"] = round(elo.rating(home), 1)
    out["elo_away_rating"] = round(elo.rating(away), 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="AWAY@HOME")
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    away, home = args.game.replace("@", " ").upper().split()
    print(json.dumps(predict(home, away, args.season), indent=2))
