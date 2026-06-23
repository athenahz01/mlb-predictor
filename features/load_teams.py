"""
features/load_teams.py
----------------------
Turn your Statcast-derived rate snapshots + a real game's pitchers/lineups into
the Team objects the simulator consumes.

Inputs it reads (all produced by ingest/pull_statcast.py):
  data/snapshots/pa_rates_batter_<season>.parquet    (index = MLBAM batter id)
  data/snapshots/pa_rates_pitcher_<season>.parquet   (index = MLBAM pitcher id)
  data/snapshots/statcast_<season>.parquet           (for handedness; cached after 1st read)

Lineups, in order of preference:
  1. CONFIRMED  - statsapi boxscore battingOrder (posts ~1-3 hrs pre-game)
  2. PROJECTED  - team's most recent posted lineup (previous game)
  3. FALLBACK   - league-average filler (flagged loudly)

A missing player (rookie / tiny sample) falls back to a league-average profile
rather than crashing. Everything degrades; nothing hard-fails.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from features.pa_probabilities import Batter, Pitcher
from sim.markov_game import Team

EVENTS = config.EVENTS
_L = config.LEAGUE_PA_RATES

# regression-to-the-mean constants (pseudo-sample of league-average play).
# A player is blended: weight = own_sample / (own_sample + K). Low-PA players
# get pulled hard toward league average, which kills flukey extremes (a 6-PA
# hitter with 2 HR is NOT a 33%-HR hitter).
K_BAT = 150     # in plate appearances
K_PIT = 200     # in batters faced


def _shrink(rates: dict, sample: int, K: int) -> dict:
    """Blend a player's per-PA rates toward league average by sample size.
    Both inputs sum to 1, so the blend stays a valid distribution."""
    w = sample / (sample + K) if sample and sample > 0 else 0.0
    return {e: w * rates.get(e, _L[e]) + (1 - w) * _L[e] for e in EVENTS}

# a slightly-better-than-league bullpen profile (relievers miss more bats)
_BULLPEN_RATES = dict(_L)
_BULLPEN_RATES["K"] = 0.245
_BULLPEN_RATES["IP_OUT"] = _L["IP_OUT"] - 0.021


# --------------------------------------------------------------------------
# Rate + handedness tables
# --------------------------------------------------------------------------
def _hand_cache(season: int) -> tuple[dict, dict]:
    """Build (or load) per-player dominant handedness from the statcast snapshot.
    Cached to a tiny parquet so we don't re-read 770k rows every run."""
    bcache = config.SNAPSHOTS / f"hand_batter_{season}.parquet"
    pcache = config.SNAPSHOTS / f"hand_pitcher_{season}.parquet"
    if bcache.exists() and pcache.exists():
        b = pd.read_parquet(bcache); p = pd.read_parquet(pcache)
        return (dict(zip(b["id"], b["hand"])), dict(zip(p["id"], p["hand"])))

    sc = pd.read_parquet(config.SNAPSHOTS / f"statcast_{season}.parquet",
                         columns=["batter", "pitcher", "stand", "p_throws"])
    # batter: switch if both L and R appear meaningfully, else the mode
    bh = {}
    for bid, grp in sc.groupby("batter")["stand"]:
        share = grp.value_counts(normalize=True)
        if len(share) > 1 and share.min() > 0.20:
            bh[int(bid)] = "S"
        else:
            bh[int(bid)] = str(share.idxmax())
    ph = {int(pid): str(g.value_counts().idxmax())
          for pid, g in sc.groupby("pitcher")["p_throws"]}

    pd.DataFrame({"id": list(bh), "hand": list(bh.values())}).to_parquet(bcache)
    pd.DataFrame({"id": list(ph), "hand": list(ph.values())}).to_parquet(pcache)
    return bh, ph


def load_rate_tables(season: int):
    bat = pd.read_parquet(config.SNAPSHOTS / f"pa_rates_batter_{season}.parquet")
    pit = pd.read_parquet(config.SNAPSHOTS / f"pa_rates_pitcher_{season}.parquet")
    bhand, phand = _hand_cache(season)
    return {"bat": bat, "pit": pit, "bhand": bhand, "phand": phand}


# --------------------------------------------------------------------------
# Build player objects from ids
# --------------------------------------------------------------------------
def _rates_row(pid: int, table: pd.DataFrame) -> Optional[dict]:
    if pid in table.index:
        row = table.loc[pid]
        return {e: float(row[e]) for e in EVENTS}, int(row.get("PA", 0))
    return None


def make_batter(pid: int, name: str, tables: dict, order: int) -> Batter:
    got = _rates_row(pid, tables["bat"])
    if got:
        rates = _shrink(got[0], got[1], K_BAT)
    else:
        rates = dict(_L)
    hand = tables["bhand"].get(int(pid), "R")
    return Batter(name=name, rates=rates, hand=hand, order=order)


def make_pitcher(pid: int, name: str, tables: dict, is_starter=True) -> Pitcher:
    got = _rates_row(pid, tables["pit"])
    if got:
        rates = _shrink(got[0], got[1], K_PIT)
    else:
        rates = dict(_L)
    hand = tables["phand"].get(int(pid), "R")
    return Pitcher(name=name, rates=rates, hand=hand, is_starter=is_starter)


def make_bullpen(code: str) -> Pitcher:
    return Pitcher(name=f"{code}_pen", rates=dict(_BULLPEN_RATES), hand="R",
                   is_starter=False)


def build_team(code: str, lineup: list[tuple[int, str]], starter_id: int,
               starter_name: str, tables: dict) -> Team:
    """lineup = list of 9 (mlbam_id, name) in batting order."""
    batters = [make_batter(pid, nm, tables, order=i + 1)
               for i, (pid, nm) in enumerate(lineup)]
    while len(batters) < 9:                      # pad if a short lineup slipped through
        batters.append(make_batter(-1, f"{code}_filler{len(batters)+1}",
                                    tables, order=len(batters) + 1))
    starter = make_pitcher(starter_id, starter_name, tables, is_starter=True)
    return Team(code=code, lineup=batters[:9], starter=starter,
                bullpen=make_bullpen(code))


# --------------------------------------------------------------------------
# Schedule + lineup resolution (statsapi)
# --------------------------------------------------------------------------
def load_schedule(date: str) -> list[dict]:
    path = config.SNAPSHOTS / f"schedule_{date}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No schedule snapshot for {date}. Run:\n"
            f"  python -m ingest.pull_mlb_statsapi --date {date}")
    return json.loads(path.read_text())["games"]


def find_game(games: list[dict], spec: str) -> dict:
    """spec like 'LAD@COL' (away@home), case-insensitive."""
    away, home = [s.strip().upper() for s in spec.replace("@", " ").split()]
    for g in games:
        if g["away"].upper() == away and g["home"].upper() == home:
            return g
    have = ", ".join(f'{g["away"]}@{g["home"]}' for g in games)
    raise ValueError(f"{spec} not in schedule. Today: {have}")


def resolve_lineup(gamePk: int, side: str) -> tuple[list[tuple[int, str]], str]:
    """
    Return (lineup, source). Tries confirmed boxscore order first; on failure
    or empty, returns ([], 'none') so the caller can decide.
    """
    try:
        from ingest.pull_mlb_statsapi import boxscore_lineups
        box = boxscore_lineups(gamePk)
        order = box.get(side, [])
        lineup = [(p["id"], p["name"]) for p in order if p.get("id")]
        if len(lineup) >= 9:
            return lineup[:9], "confirmed"
    except Exception as e:
        print(f"[load] boxscore lookup failed for {side} ({type(e).__name__})")
    return [], "none"


def projected_lineup_from_roster(team_id: int, tables: dict,
                                 n: int = 9) -> list[tuple[int, str]]:
    """
    Fallback projected lineup: active hitters on the 40-man, top-N by PA in the
    rate table, ordered by PA (a crude proxy, NOT a real batting order).
    """
    try:
        import statsapi
        roster = statsapi.get("team_roster",
                              {"teamId": team_id, "rosterType": "active"})
        ids = {}
        for p in roster.get("roster", []):
            pid = p["person"]["id"]; nm = p["person"]["fullName"]
            pos = p.get("position", {}).get("abbreviation", "")
            if pos not in ("P",):                 # hitters only
                ids[pid] = nm
        bat = tables["bat"]
        cand = [(pid, nm, int(bat.loc[pid, "PA"]))
                for pid, nm in ids.items() if pid in bat.index]
        cand.sort(key=lambda x: -x[2])
        return [(pid, nm) for pid, nm, _ in cand[:n]]
    except Exception as e:
        print(f"[load] projected-lineup fallback failed ({type(e).__name__})")
        return []


def build_game(spec: str, date: str, season: int):
    """
    Full path: snapshot schedule -> find game -> resolve both lineups ->
    build Team objects + GameContext. Returns (home, away, ctx, info).
    """
    from sim.markov_game import GameContext
    games = load_schedule(date)
    g = find_game(games, spec)
    tables = load_rate_tables(season)

    info = {"game": f'{g["away"]}@{g["home"]}', "venue": g.get("venue"),
            "home_sp": g.get("home_probable"), "away_sp": g.get("away_probable")}

    sources = {}
    sides = {}
    for side, code, tid in (("home", g["home"], g["home_team_id"]),
                            ("away", g["away"], g["away_team_id"])):
        lineup, src = resolve_lineup(g["gamePk"], side)
        if not lineup:
            lineup = projected_lineup_from_roster(tid, tables)
            src = "projected" if lineup else "fallback"
        sources[side] = src
        sides[side] = (code, lineup)
    info["lineup_source"] = sources

    home = build_team(sides["home"][0], sides["home"][1],
                      g.get("home_probable_id") or -1,
                      g.get("home_probable") or "TBD", tables)
    away = build_team(sides["away"][0], sides["away"][1],
                      g.get("away_probable_id") or -1,
                      g.get("away_probable") or "TBD", tables)

    ctx = GameContext(park_code=g["home"])        # home team abbr -> park factor key
    return home, away, ctx, info
