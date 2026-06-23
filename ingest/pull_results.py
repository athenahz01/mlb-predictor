"""
ingest/pull_results.py
----------------------
Pull completed game results for a season from the MLB Stats API, for building
the Elo + Pythagenpat baselines. Saves a snapshot and a per-team run aggregate.

  python -m ingest.pull_results --season 2026

Output (data/snapshots/):
  results_<season>.json       chronological list of finished games + scores
  team_runs_<season>.json     per-team {R, RA, G} for Pythagenpat
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict

import requests

import config

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def pull(season: int) -> list[dict]:
    start = f"{season}-03-01"
    end = dt.date.today().isoformat()
    url = f"{config.STATSAPI_BASE}/schedule"
    params = {"sportId": 1, "startDate": start, "endDate": end,
              "hydrate": "team,linescore,probablePitcher"}
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    games = []
    skipped_tie = 0
    skipped_type = 0
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            # regular season only: drop spring training (S), exhibitions/WBC (E/I),
            # All-Star (A). These have non-representative lineups (rested stars,
            # national teams, minor leaguers) and meaningless outcomes.
            if g.get("gameType") != "R":
                skipped_type += 1
                continue
            if g["status"]["abstractGameState"] != "Final":
                continue
            h, a = g["teams"]["home"], g["teams"]["away"]

            # canonical score = linescore runs; fall back to teams.score
            ls = g.get("linescore", {}).get("teams", {})
            hr = ls.get("home", {}).get("runs", h.get("score"))
            ar = ls.get("away", {}).get("runs", a.get("score"))
            if hr is None or ar is None:
                continue
            if hr == ar:                          # no MLB final is a tie -> bad/suspended
                skipped_tie += 1
                continue

            # cross-check against the authoritative isWinner flag; trust it on conflict
            hw = h.get("isWinner")
            if hw is not None and (hr > ar) != hw:
                hr, ar = ar, hr                   # scores were backwards; flip to match winner

            games.append({
                "date": g["gameDate"][:10],
                "gamePk": g["gamePk"],
                "home": h["team"].get("abbreviation", h["team"]["name"]),
                "away": a["team"].get("abbreviation", a["team"]["name"]),
                "home_score": hr,
                "away_score": ar,
            })
    games.sort(key=lambda x: (x["date"], x["gamePk"]))
    if skipped_type:
        print(f"[results] skipped {skipped_type} non-regular-season games (spring/exhibition/WBC)")
    if skipped_tie:
        print(f"[results] skipped {skipped_tie} games with tied/invalid scores")
    return games


def team_aggregates(games: list[dict]) -> dict:
    agg = defaultdict(lambda: {"R": 0, "RA": 0, "G": 0})
    for g in games:
        agg[g["home"]]["R"] += g["home_score"]; agg[g["home"]]["RA"] += g["away_score"]; agg[g["home"]]["G"] += 1
        agg[g["away"]]["R"] += g["away_score"]; agg[g["away"]]["RA"] += g["home_score"]; agg[g["away"]]["G"] += 1
    return dict(agg)


def snapshot(season: int):
    games = pull(season)
    (config.SNAPSHOTS / f"results_{season}.json").write_text(json.dumps(games, indent=2))
    agg = team_aggregates(games)
    (config.SNAPSHOTS / f"team_runs_{season}.json").write_text(json.dumps(agg, indent=2))
    print(f"[results] {len(games)} final games -> results_{season}.json")
    print(f"[results] {len(agg)} teams -> team_runs_{season}.json")
    return games, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    snapshot(args.season)
