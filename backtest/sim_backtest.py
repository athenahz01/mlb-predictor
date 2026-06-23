"""
backtest/sim_backtest.py
------------------------
Put the MAIN predictor (the simulation) on trial against the baselines.

For every completed game we:
  1. pull its actual batting orders + starters from the boxscore,
  2. build teams from the rate snapshots, run the simulation -> P(home win),
  3. compare against Elo and Pythagenpat predictions computed walk-forward
     (leakage-free: each baseline only sees games before the one it predicts),
  4. paired-bootstrap the sim's log-loss vs each baseline -> ship/hold verdict.

This is the test the content series teased: does the 20,000-game simulation
actually beat the simple measuring sticks, or not? Either answer is honest.

Sim predictions are cached to data/snapshots/sim_backtest_<season>.json so you
can stop and resume without re-simulating games already done.

  python -m backtest.sim_backtest --season 2026 --rate-season 2025 --sims 2000 --limit 150
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import requests

import config
from models.elo import EloModel
from models import pythag
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation
from backtest.walk_forward import paired_bootstrap_pvalue, logloss

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# --------------------------------------------------------------------------
# leakage-free baseline predictions over the full chronological sequence
# --------------------------------------------------------------------------
def walk_forward_baselines(games: list[dict]) -> dict:
    """gamePk -> {'elo_p', 'pythag_p', 'y'} computed using only prior games."""
    elo = EloModel()
    runs = {}
    out = {}
    for g in games:
        h, a = g["home"], g["away"]
        hp, ap = runs.get(h), runs.get(a)
        if hp and ap and hp["G"] > 0 and ap["G"] > 0:
            pythag_p = pythag.predict(hp, ap)
        else:
            pythag_p = 0.535            # neutral prior with small home edge
        out[g["gamePk"]] = {
            "elo_p": elo.predict(h, a),
            "pythag_p": pythag_p,
            "y": int(g["home_score"] > g["away_score"]),
        }
        elo.update(h, a, g["home_score"], g["away_score"])
        for team, rf, ra in ((h, g["home_score"], g["away_score"]),
                             (a, g["away_score"], g["home_score"])):
            r = runs.setdefault(team, {"R": 0, "RA": 0, "G": 0})
            r["R"] += rf; r["RA"] += ra; r["G"] += 1
    return out


# --------------------------------------------------------------------------
# pull actual lineups + starters for a completed game
# --------------------------------------------------------------------------
def fetch_lineups_starters(gamePk: int):
    """Return {'home': (lineup, starter), 'away': (lineup, starter)} or None."""
    url = f"{config.STATSAPI_BASE}/game/{gamePk}/boxscore"
    try:
        d = SESSION.get(url, timeout=20).json()
    except Exception:
        return None
    out = {}
    for side in ("home", "away"):
        team = d["teams"][side]
        players = team.get("players", {})
        order = team.get("battingOrder", [])
        lineup = []
        for pid in order:
            p = players.get(f"ID{pid}", {})
            lineup.append((pid, p.get("person", {}).get("fullName", str(pid))))
        pitchers = team.get("pitchers", [])
        if pitchers:
            sp_id = pitchers[0]
            sp = players.get(f"ID{sp_id}", {})
            starter = (sp_id, sp.get("person", {}).get("fullName", str(sp_id)))
        else:
            starter = (-1, "TBD")
        if len(lineup) < 9:
            return None
        out[side] = (lineup[:9], starter)
    return out


# --------------------------------------------------------------------------
# main backtest
# --------------------------------------------------------------------------
def run(season: int, rate_season: int, n_sims: int = 2000,
        limit: int | None = None, fetch_fn=fetch_lineups_starters) -> dict:
    rpath = config.SNAPSHOTS / f"results_{season}.json"
    if not rpath.exists():
        raise FileNotFoundError(
            f"No results. Run: python -m ingest.pull_results --season {season}")
    games = json.loads(rpath.read_text())
    games.sort(key=lambda g: (g["date"], g["gamePk"]))

    baselines = walk_forward_baselines(games)      # over the full sequence
    tables = load_rate_tables(rate_season)

    cache_path = config.SNAPSHOTS / f"sim_backtest_{season}.json"
    raw_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # cache stores ONLY sim_p (the expensive part). Old caches stored a dict with
    # baselines+outcome baked in; normalize so we reuse the sim_p but recompute
    # baselines/outcomes fresh from (possibly corrected) results.
    sim_cache = {pk: (v["sim_p"] if isinstance(v, dict) else v)
                 for pk, v in raw_cache.items()}

    eval_games = games[:limit] if limit else games
    done_new = 0
    for g in eval_games:
        pk = str(g["gamePk"])
        if pk in sim_cache:
            continue
        lu = fetch_fn(g["gamePk"])
        if not lu:
            continue
        (h_line, h_sp), (a_line, a_sp) = lu["home"], lu["away"]
        home = build_team(g["home"], h_line, h_sp[0], h_sp[1], tables)
        away = build_team(g["away"], a_line, a_sp[0], a_sp[1], tables)
        ctx = GameContext(park_code=g["home"])
        res = run_simulation(home, away, ctx, n_sims=n_sims, seed=g["gamePk"] % 9999)
        sim_cache[pk] = res["p_home_win"]
        done_new += 1
        if done_new % 25 == 0:
            cache_path.write_text(json.dumps(sim_cache))
            print(f"  ...simulated {done_new} new games")
    cache_path.write_text(json.dumps(sim_cache))

    # assemble evaluation arrays: cached sim_p + FRESH baselines/outcome
    rows = []
    for g in eval_games:
        pk = str(g["gamePk"])
        if pk in sim_cache and g["gamePk"] in baselines:
            rows.append({"sim_p": sim_cache[pk], **baselines[g["gamePk"]]})
    if not rows:
        return {"n": 0, "note": "no games simulated yet"}
    sim_p = np.array([r["sim_p"] for r in rows])
    elo_p = np.array([r["elo_p"] for r in rows])
    pyth_p = np.array([r["pythag_p"] for r in rows])
    y = np.array([r["y"] for r in rows], float)

    L_sim, L_elo, L_pyth = logloss(sim_p, y), logloss(elo_p, y), logloss(pyth_p, y)
    p_vs_elo = paired_bootstrap_pvalue(L_sim, L_elo)
    p_vs_pyth = paired_bootstrap_pvalue(L_sim, L_pyth)

    return {
        "n": len(rows),
        "sim_logloss": float(L_sim.mean()),
        "elo_logloss": float(L_elo.mean()),
        "pythag_logloss": float(L_pyth.mean()),
        "sim_acc": float(np.mean((sim_p > 0.5) == (y == 1))),
        "elo_acc": float(np.mean((elo_p > 0.5) == (y == 1))),
        "p_sim_vs_elo": p_vs_elo,
        "p_sim_vs_pythag": p_vs_pyth,
        "sim_beats_elo": bool(p_vs_elo < 0.05),
        "sim_beats_pythag": bool(p_vs_pyth < 0.05),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--rate-season", type=int, default=dt.date.today().year - 1)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    r = run(args.season, args.rate_season, args.sims, args.limit)
    if r.get("n", 0) == 0:
        print(r.get("note", "nothing to report")); raise SystemExit
    print(f"\nSim on trial over {r['n']} completed games:")
    print(f"  simulation   log-loss {r['sim_logloss']:.4f}   acc {r['sim_acc']:.3f}")
    print(f"  Elo          log-loss {r['elo_logloss']:.4f}   acc {r['elo_acc']:.3f}")
    print(f"  Pythagenpat  log-loss {r['pythag_logloss']:.4f}")
    print(f"\n  paired-bootstrap p, sim vs Elo:        {r['p_sim_vs_elo']:.4f}")
    print(f"  paired-bootstrap p, sim vs Pythagenpat: {r['p_sim_vs_pythag']:.4f}")
    print()
    for name, beats in (("Elo", r["sim_beats_elo"]), ("Pythagenpat", r["sim_beats_pythag"])):
        print(f"  {'PASS' if beats else 'HOLD'} - simulation",
              "beats" if beats else "does NOT beat", name, "at p<0.05")
