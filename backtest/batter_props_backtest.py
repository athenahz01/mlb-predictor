"""
batter_props_backtest.py
------------------------
Put the PER-BATTER props on trial. The sim emits, for every hitter in the lineup,
a P(HR) and an expected total-base count. This replays completed games, builds
each game from its real boxscore lineup + starter (prior-season rates, so it's
leakage-free), runs the sim, and checks whether those per-batter numbers are
calibrated against what each hitter actually did.

  python -m backtest.batter_props_backtest --season 2026 --rate-season 2025 --sims 2000 --limit 100

Reports:
  HR  -> predicted P(HR) vs actual HR rate, reliability curve + ECE
  TB  -> predicted E[TB] vs actual total bases, bias + MAE
Bullpen is left league-average here (no current-roster lookup) so the backtest
stays point-in-time clean.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation
from backtest.props_backtest import reliability, _fmt_reliability, SESSION


def fetch_batters(game_pk: int):
    """One boxscore fetch -> per side: lineup [(id,name)], starter (id,name),
    and per-batter actuals {pid: (homered 0/1, total_bases)}."""
    url = f"{config.STATSAPI_BASE}/game/{game_pk}/boxscore"
    try:
        box = SESSION.get(url, timeout=20).json()
    except Exception:
        return None
    out = {"lineups": {}, "starters": {}, "actual": {}}
    for side in ("home", "away"):
        team = box["teams"][side]
        players = team.get("players", {})
        order = team.get("battingOrder", [])
        if len(order) < 9:
            return None
        lineup, actual = [], {}
        for pid in order[:9]:
            p = players.get(f"ID{pid}", {})
            lineup.append((pid, p.get("person", {}).get("fullName", str(pid))))
            bat = p.get("stats", {}).get("batting", {})
            hr = int(bat.get("homeRuns", 0) or 0)
            tb = bat.get("totalBases")
            if tb is None:                       # 1B+2*2B+3*3B+4*HR = hits+2B+2*3B+3*HR
                h = int(bat.get("hits", 0) or 0)
                d = int(bat.get("doubles", 0) or 0)
                t = int(bat.get("triples", 0) or 0)
                tb = h + d + 2 * t + 3 * hr
            actual[pid] = (1 if hr > 0 else 0, int(tb))
        pitchers = team.get("pitchers", [])
        if not pitchers:
            return None
        sp_id = pitchers[0]
        sp = players.get(f"ID{sp_id}", {})
        out["lineups"][side] = lineup
        out["starters"][side] = (sp_id, sp.get("person", {}).get("fullName", str(sp_id)))
        out["actual"][side] = actual
    return out


def run(season: int, rate_season: int, n_sims: int = 2000, limit: int = 100,
        recent: bool = False):
    results = json.loads((config.SNAPSHOTS / f"results_{season}.json").read_text())
    results = [g for g in results if g.get("home_score") is not None]
    results = results[-limit:] if recent else results[:limit]
    tables = load_rate_tables(rate_season)

    p_hr, hr_act, tb_pred, tb_act = [], [], [], []
    done = 0
    for g in results:
        fb = fetch_batters(g["gamePk"])
        if not fb:
            continue
        home = build_team(g["home"], fb["lineups"]["home"],
                          fb["starters"]["home"][0], fb["starters"]["home"][1], tables)
        away = build_team(g["away"], fb["lineups"]["away"],
                          fb["starters"]["away"][0], fb["starters"]["away"][1], tables)
        res = run_simulation(home, away, GameContext(park_code=g["home"]),
                             n_sims=n_sims, seed=g["gamePk"] % 9999)
        for side, key in (("home", "home_batters"), ("away", "away_batters")):
            preds = res[key]
            for i, (pid, _) in enumerate(fb["lineups"][side]):
                if pid not in fb["actual"][side]:
                    continue
                a_hr, a_tb = fb["actual"][side][pid]
                p_hr.append(preds[i]["p_hr"]); hr_act.append(a_hr)
                tb_pred.append(preds[i]["exp_tb"]); tb_act.append(a_tb)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done} games, {len(p_hr)} batter-games")

    p = np.array(p_hr); h = np.array(hr_act)
    tp = np.array(tb_pred); ta = np.array(tb_act, float)
    print(f"\nPer-batter props over {done} games ({len(p)} batter-games):")

    hr_bias = p.mean() - h.mean()
    rel, ece = reliability(p, h, n_bins=5)
    print(f"HOME-RUN  pred {p.mean():.3f} vs actual {h.mean():.3f} (bias {hr_bias:+.3f})")
    print(f"  reliability {_fmt_reliability(rel)}  ECE {ece:.3f}")
    print(f"  => {'READY' if abs(hr_bias) < 0.01 and ece < 0.03 else 'NEEDS WORK'}")

    tb_bias = tp.mean() - ta.mean()
    print(f"TOTAL BASES  pred {tp.mean():.3f} vs actual {ta.mean():.3f} "
          f"(bias {tb_bias:+.3f}, MAE {np.abs(tp - ta).mean():.3f})")
    print(f"  => {'READY' if abs(tb_bias) < 0.10 else 'NEEDS WORK'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--rate-season", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--recent", action="store_true",
                    help="use the most recent games (summer) instead of the earliest")
    a = ap.parse_args()
    run(a.season, a.rate_season, a.sims, a.limit, a.recent)