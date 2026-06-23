"""
ingest/pull_mlb_statsapi.py
---------------------------
Official MLB Stats API (statsapi.mlb.com) - free, no key. This is the
OPERATIONAL backbone: schedule, probable pitchers, and confirmed lineups.

The single most important discipline carryover from your NHL/NBA pipelines:
LISTED-PITCHER CONFIRMATION. One starter drives 30-40% of run-scoring, so a
late scratch invalidates the whole matchup premise. Poll this in the hours
before first pitch and re-run predictions on any change. Same rule as your
lineup-confirmation discipline, but even higher stakes.

Usage:
  python -m ingest.pull_mlb_statsapi --date 2026-04-19
  python -m ingest.pull_mlb_statsapi --probables --date today
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import requests

import config

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _date(s: str) -> str:
    if s in ("today", None):
        return dt.date.today().isoformat()
    return s


def schedule(date: str, sportId: int = 1) -> list[dict]:
    """Games for a date, with probable pitchers hydrated."""
    url = f"{config.STATSAPI_BASE}/schedule"
    params = {"sportId": sportId, "date": date,
              "hydrate": "probablePitcher,team,linescore"}
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]; away = g["teams"]["away"]
            games.append({
                "gamePk": g["gamePk"],
                "gameDate": g["gameDate"],
                "status": g["status"]["detailedState"],
                "home": home["team"]["abbreviation"]
                        if "abbreviation" in home["team"] else home["team"]["name"],
                "away": away["team"]["abbreviation"]
                        if "abbreviation" in away["team"] else away["team"]["name"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_probable": (home.get("probablePitcher") or {}).get("fullName"),
                "away_probable": (away.get("probablePitcher") or {}).get("fullName"),
                "home_probable_id": (home.get("probablePitcher") or {}).get("id"),
                "away_probable_id": (away.get("probablePitcher") or {}).get("id"),
                "venue": g.get("venue", {}).get("name"),
            })
    return games


def boxscore_lineups(gamePk: int) -> dict:
    """Confirmed batting orders once posted (empty until lineups drop)."""
    url = f"{config.STATSAPI_BASE}/game/{gamePk}/boxscore".replace("/v1/", "/v1.1/")
    # boxscore lives on the v1 endpoint:
    url = f"{config.STATSAPI_BASE}/game/{gamePk}/boxscore"
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = {}
    for side in ("home", "away"):
        team = data["teams"][side]
        order = team.get("battingOrder", [])
        players = team.get("players", {})
        lineup = []
        for pid in order:
            p = players.get(f"ID{pid}", {})
            person = p.get("person", {})
            lineup.append({"id": person.get("id"),
                           "name": person.get("fullName"),
                           "position": p.get("position", {}).get("abbreviation")})
        out[side] = lineup
    return out


def snapshot_probables(date: str):
    date = _date(date)
    games = schedule(date)
    path = config.SNAPSHOTS / f"schedule_{date}.json"
    path.write_text(json.dumps({"date": date, "games": games}, indent=2))
    missing = [f'{g["away"]}@{g["home"]}' for g in games
               if not g["home_probable"] or not g["away_probable"]]
    print(f"[statsapi] {len(games)} games -> {path.name}")
    if missing:
        print(f"[statsapi] probable pitcher TBD for: {', '.join(missing)}")
    return games


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    ap.add_argument("--probables", action="store_true")
    args = ap.parse_args()
    snapshot_probables(args.date)
