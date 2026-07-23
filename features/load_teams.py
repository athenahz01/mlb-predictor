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


def load_rate_tables(season: int, xrates: bool = False):
    """xrates=True loads the DELUCKED contact-quality tables (ingest/xrates.py)."""
    stem = "pa_xrates" if xrates else "pa_rates"
    bat = pd.read_parquet(config.SNAPSHOTS / f"{stem}_batter_{season}.parquet")
    pit = pd.read_parquet(config.SNAPSHOTS / f"{stem}_pitcher_{season}.parquet")
    bhand, phand = _hand_cache(season)
    return {"bat": bat, "pit": pit, "bhand": bhand, "phand": phand}


def _blend_tables(cur: pd.DataFrame, prior: pd.DataFrame,
                  prior_weight: int) -> pd.DataFrame:
    """
    Blend current-season rates with prior-season rates per player, weighting the
    prior as `prior_weight` pseudo-PAs. Early in the season (little current data)
    the estimate leans on last year; as current PA accrues it takes over. Players
    in only one season keep that season's rates. Combined PA is carried so the
    downstream league-shrink still regresses thin samples correctly.
    """
    cols = EVENTS
    out = {}
    idx = cur.index.union(prior.index)
    for pid in idx:
        in_cur = pid in cur.index
        in_prior = pid in prior.index
        if in_cur and in_prior:
            rc = cur.loc[pid]; pc = int(rc.get("PA", 0))
            rp = prior.loc[pid]
            w = min(int(rp.get("PA", 0)), prior_weight)
            denom = pc + w if (pc + w) > 0 else 1
            row = {e: (pc * float(rc[e]) + w * float(rp[e])) / denom for e in cols}
            row["PA"] = pc + w
        elif in_cur:
            r = cur.loc[pid]; row = {e: float(r[e]) for e in cols}
            row["PA"] = int(r.get("PA", 0))
        else:
            r = prior.loc[pid]; row = {e: float(r[e]) for e in cols}
            row["PA"] = int(r.get("PA", 0))
        out[pid] = row
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = cur.index.name
    return df


def load_blended_rate_tables(cur_season: int, prior_season: int,
                             prior_weight: int = 200, xrates: bool = False):
    """
    LIVE prediction tables: current-season rates blended onto last season.
    Use this for predicting TODAY's games (current-season-to-date is real past
    data, so no leakage). Do NOT use for historical backtests -- that leaks
    future games into past predictions; use load_rate_tables(prior) there.
    """
    cur = load_rate_tables(cur_season, xrates=xrates)
    prior = load_rate_tables(prior_season, xrates=xrates)
    bat = _blend_tables(cur["bat"], prior["bat"], prior_weight)
    pit = _blend_tables(cur["pit"], prior["pit"], prior_weight)
    bhand = {**prior["bhand"], **cur["bhand"]}   # prefer current-season handedness
    phand = {**prior["phand"], **cur["phand"]}
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
    flags = []
    if got:
        rates = _shrink(got[0], got[1], K_BAT)
    else:
        rates = dict(_L)
        flags.append("missing_batter_rates_league_prior")
    hand = tables["bhand"].get(int(pid))
    if hand is None:
        hand = "R"
        flags.append("missing_batter_handedness_assumed_right")
    return Batter(
        name=name,
        rates=rates,
        hand=hand,
        order=order,
        mlb_id=pid if pid > 0 else None,
        data_quality_flags=tuple(flags),
    )


def make_pitcher(pid: int, name: str, tables: dict, is_starter=True) -> Pitcher:
    got = _rates_row(pid, tables["pit"])
    flags = []
    if got:
        rates = _shrink(got[0], got[1], K_PIT)
    else:
        rates = dict(_L)
        flags.append("missing_pitcher_rates_league_prior")
    hand = tables["phand"].get(int(pid))
    if hand is None:
        hand = "R"
        flags.append("missing_pitcher_handedness_assumed_right")
    return Pitcher(
        name=name,
        rates=rates,
        hand=hand,
        is_starter=is_starter,
        mlb_id=pid if pid > 0 else None,
        data_quality_flags=tuple(flags),
    )


def make_bullpen(code: str) -> Pitcher:
    """League-average bullpen profile (fallback when no roster/rates available)."""
    return Pitcher(
        name=f"{code}_pen",
        rates=dict(_BULLPEN_RATES),
        hand="R",
        is_starter=False,
        data_quality_flags=("bullpen_league_average_fallback",),
    )


# per-team bullpen: aggregate the team's actual relievers, usage-weighted,
# shrunk toward the league bullpen profile so thin pens stay stable.
RELIEVER_MIN_PA = 40        # enough batters faced for the rate to mean something
RELIEVER_MAX_PA = 450       # above this it's a starter's workload, not a relief arm
BULLPEN_SHRINK_K = 200      # pseudo-PAs of league prior mixed into the team aggregate


def _team_pitcher_ids(team_id: int) -> list[int]:
    import requests
    url = f"{config.STATSAPI_BASE}/teams/{team_id}/roster"
    roster = requests.get(url, params={"rosterType": "active"}, timeout=20).json()
    return [p["person"]["id"] for p in roster.get("roster", [])
            if p.get("position", {}).get("abbreviation") == "P"]


def make_team_bullpen(code: str, team_id: int, tables: dict,
                      starter_id: int = -1,
                      unavailable: dict | None = None) -> Pitcher:
    """Build a bullpen profile from the team's relief arms (everyone but the day's
    starter, in the relief-workload PA band), usage-weighted by batters faced and
    regressed to the league bullpen prior. Falls back to league-average if the
    roster/rates aren't available."""
    try:
        pids = _team_pitcher_ids(team_id)
    except Exception:
        return make_bullpen(code)

    pit = tables["pit"]
    relievers = []
    for pid in pids:
        if pid == starter_id:
            continue
        got = _rates_row(pid, pit)
        if not got:
            continue
        rates, pa = got
        if RELIEVER_MIN_PA <= pa <= RELIEVER_MAX_PA:
            w = (unavailable or {}).get(pid, 1.0)   # recent-usage downweight
            if w > 0:
                relievers.append((rates, pa * w))

    if not relievers:
        return make_bullpen(code)

    total_pa = sum(pa for _, pa in relievers)
    agg = {e: sum(r[e] * pa for r, pa in relievers) / total_pa for e in EVENTS}
    w = total_pa / (total_pa + BULLPEN_SHRINK_K)          # regress to league prior
    rates = {e: w * agg[e] + (1 - w) * _BULLPEN_RATES[e] for e in EVENTS}
    return Pitcher(
        name=f"{code}_pen",
        rates=rates,
        hand="R",
        is_starter=False,
        data_quality_flags=("bullpen_aggregate_not_role_aware",),
    )


def build_team(code: str, lineup: list[tuple[int, str]], starter_id: int,
               starter_name: str, tables: dict, team_id: int = None,
               unavailable: dict | None = None) -> Team:
    """lineup = list of 9 (mlbam_id, name) in batting order."""
    batters = [make_batter(pid, nm, tables, order=i + 1)
               for i, (pid, nm) in enumerate(lineup)]
    while len(batters) < 9:                      # pad if a short lineup slipped through
        batters.append(make_batter(-1, f"{code}_filler{len(batters)+1}",
                                    tables, order=len(batters) + 1))
    starter = make_pitcher(starter_id, starter_name, tables, is_starter=True)
    pen = (make_team_bullpen(code, team_id, tables, starter_id,
                             unavailable=unavailable)
           if team_id else make_bullpen(code))
    return Team(code=code, lineup=batters[:9], starter=starter, bullpen=pen)


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
        import requests
        url = f"{config.STATSAPI_BASE}/teams/{team_id}/roster"
        roster = requests.get(url, params={"rosterType": "active"}, timeout=20).json()
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


def build_game(spec: str, date: str, season: int, tables=None,
               env: bool = False, workload: bool = False,
               availability: bool = False):
    """
    Full path: snapshot schedule -> find game -> resolve both lineups ->
    build Team objects + GameContext. Returns (home, away, ctx, info).
    Pass `tables` (e.g. from load_blended_rate_tables) to override the rate source.
    env         : apply game-day weather + umpire multipliers (live only).
    workload    : per-starter pitch limits from recent pitch counts (live only).
    availability: downweight relievers used the previous two days (live only).
    """
    from sim.markov_game import GameContext
    games = load_schedule(date)
    g = find_game(games, spec)
    if tables is None:
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

    unavail_h = unavail_a = None
    if availability:
        from features.environment import unavailable_relievers
        if g.get("home_team_id"):
            unavail_h = unavailable_relievers(g["home_team_id"], date)
        if g.get("away_team_id"):
            unavail_a = unavailable_relievers(g["away_team_id"], date)
    home = build_team(sides["home"][0], sides["home"][1],
                      g.get("home_probable_id") or -1,
                      g.get("home_probable") or "TBD", tables,
                      team_id=g.get("home_team_id"), unavailable=unavail_h)
    away = build_team(sides["away"][0], sides["away"][1],
                      g.get("away_probable_id") or -1,
                      g.get("away_probable") or "TBD", tables,
                      team_id=g.get("away_team_id"), unavailable=unavail_a)
    if workload:
        from features.environment import starter_pitch_limit
        for team, pid_key in ((home, "home_probable_id"), (away, "away_probable_id")):
            pid = g.get(pid_key)
            if pid:
                lim = starter_pitch_limit(pid)
                if lim:
                    team.pitch_limit = lim

    ctx = GameContext(park_code=g["home"])        # home team abbr -> park factor key
    if env:
        from features.environment import weather_mults, ump_k_mult
        wx = weather_mults(g["home"], date)
        ctx.env_hr, ctx.env_hit = wx["hr"], wx["hit"]
        if g.get("gamePk"):
            ctx.ump_k_mult = ump_k_mult(g["gamePk"])
        info_env = wx.get("detail")
    else:
        info_env = None
    if info_env:
        info["env"] = info_env
    return home, away, ctx, info
