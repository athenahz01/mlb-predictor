"""
backtest/props_backtest.py
--------------------------
Puts the NON-WINNER markets on trial. The simulation already emits totals,
NRFI, and starter-strikeout distributions; this checks whether those numbers
are CALIBRATED against reality -- i.e. when the sim says "62% over 8.5", does
the over actually hit ~62% of the time.

For each completed game it captures the sim's predictions, pulls the actual
total runs, first-inning runs (NRFI), and each starter's strikeouts from the
boxscore + linescore, and reports per-market calibration:

  totals    : mean abs error of expected total, bias, reliability of P(over 8.5)
  NRFI      : Brier, bias, reliability
  starter K : mean abs error, bias, reliability of P(over 5.5)

A market is only "ready to post" if it is roughly unbiased AND its probability
buckets line up with reality (low calibration error). Producing a number is not
the same as that number being trustworthy -- this is what separates them.

  python -m backtest.props_backtest --season 2026 --rate-season 2025 --sims 2000 --limit 150
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import requests

import config
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

TOTAL_LINE = 8.5
K_LINE = 5.5


# --------------------------------------------------------------------------
# one boxscore fetch -> lineups, starters, AND actual SP strikeouts
# --------------------------------------------------------------------------
def fetch_game(gamePk: int):
    box_url = f"{config.STATSAPI_BASE}/game/{gamePk}/boxscore"
    line_url = f"{config.STATSAPI_BASE}/game/{gamePk}/linescore"
    try:
        box = SESSION.get(box_url, timeout=20).json()
        line = SESSION.get(line_url, timeout=20).json()
    except Exception:
        return None

    out = {"lineups": {}, "starters": {}, "sp_k": {}}
    for side in ("home", "away"):
        team = box["teams"][side]
        players = team.get("players", {})
        order = team.get("battingOrder", [])
        lineup = [(pid, players.get(f"ID{pid}", {}).get("person", {})
                   .get("fullName", str(pid))) for pid in order]
        if len(lineup) < 9:
            return None
        pitchers = team.get("pitchers", [])
        if not pitchers:
            return None
        sp_id = pitchers[0]
        sp = players.get(f"ID{sp_id}", {})
        out["lineups"][side] = lineup[:9]
        out["starters"][side] = (sp_id, sp.get("person", {}).get("fullName", str(sp_id)))
        try:
            out["sp_k"][side] = int(sp["stats"]["pitching"]["strikeOuts"])
        except Exception:
            out["sp_k"][side] = None

    innings = line.get("innings", [])
    if innings:
        first = innings[0]
        out["inning1_runs"] = ((first.get("home", {}).get("runs", 0) or 0) +
                               (first.get("away", {}).get("runs", 0) or 0))
    else:
        out["inning1_runs"] = None
    return out


# --------------------------------------------------------------------------
# calibration helpers
# --------------------------------------------------------------------------
def reliability(probs, outcomes, n_bins=5):
    """Return (mean_pred, mean_actual, count) per bin and expected calibration error."""
    probs = np.asarray(probs, float)
    outcomes = np.asarray(outcomes, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < n_bins - 1
                                   else probs <= edges[i + 1])
        if m.sum() == 0:
            continue
        mp, ma, c = probs[m].mean(), outcomes[m].mean(), int(m.sum())
        rows.append((mp, ma, c))
        ece += (c / len(probs)) * abs(mp - ma)
    return rows, ece


def _fmt_reliability(rows):
    return "  ".join(f"[{mp:.2f}->{ma:.2f} n{c}]" for mp, ma, c in rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run(season: int, rate_season: int, n_sims: int = 2000,
        limit: int | None = None) -> dict:
    results = json.loads((config.SNAPSHOTS / f"results_{season}.json").read_text())
    results.sort(key=lambda g: (g["date"], g["gamePk"]))
    tables = load_rate_tables(rate_season)

    cache_path = config.SNAPSHOTS / f"props_backtest_{season}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    eval_games = results[:limit] if limit else results
    done = 0
    for g in eval_games:
        pk = str(g["gamePk"])
        if pk in cache:
            continue
        fg = fetch_game(g["gamePk"])
        if not fg:
            continue
        home = build_team(g["home"], fg["lineups"]["home"],
                          fg["starters"]["home"][0], fg["starters"]["home"][1], tables)
        away = build_team(g["away"], fg["lineups"]["away"],
                          fg["starters"]["away"][0], fg["starters"]["away"][1], tables)
        res = run_simulation(home, away, GameContext(park_code=g["home"]),
                             n_sims=n_sims, seed=g["gamePk"] % 9999)
        cache[pk] = {
            # predictions
            "exp_total": res["exp_total"],
            "p_over_total": res["total_over"][f"over_{TOTAL_LINE}"],
            "p_nrfi": res["p_nrfi"],
            "home_sp_k_pred": res["home_starter_k"]["mean"],
            "away_sp_k_pred": res["away_starter_k"]["mean"],
            "p_home_sp_over": res["home_starter_k"]["over"][f"over_{K_LINE}"],
            "p_away_sp_over": res["away_starter_k"]["over"][f"over_{K_LINE}"],
            # actuals
            "total_actual": g["home_score"] + g["away_score"],
            "inning1_runs": fg["inning1_runs"],
            "home_sp_k_actual": fg["sp_k"]["home"],
            "away_sp_k_actual": fg["sp_k"]["away"],
        }
        done += 1
        if done % 25 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  ...simulated {done} new games")
    cache_path.write_text(json.dumps(cache))

    rows = [cache[str(g["gamePk"])] for g in eval_games if str(g["gamePk"]) in cache]
    if not rows:
        return {"n": 0}

    out = {"n": len(rows)}

    # ---- totals ----
    et = np.array([r["exp_total"] for r in rows])
    at = np.array([r["total_actual"] for r in rows])
    p_over = np.array([r["p_over_total"] for r in rows])
    over_hit = (at > TOTAL_LINE).astype(float)
    rel_t, ece_t = reliability(p_over, over_hit)
    out["totals"] = {
        "mae": float(np.mean(np.abs(et - at))),
        "pred_mean": float(et.mean()), "actual_mean": float(at.mean()),
        "bias": float(et.mean() - at.mean()),
        "over_line": TOTAL_LINE, "pred_over_rate": float(p_over.mean()),
        "actual_over_rate": float(over_hit.mean()),
        "ece": ece_t, "reliability": rel_t,
    }

    # ---- NRFI ----
    nrfi_rows = [r for r in rows if r["inning1_runs"] is not None]
    if nrfi_rows:
        p_nrfi = np.array([r["p_nrfi"] for r in nrfi_rows])
        nrfi_act = np.array([int(r["inning1_runs"] == 0) for r in nrfi_rows], float)
        rel_n, ece_n = reliability(p_nrfi, nrfi_act)
        out["nrfi"] = {
            "n": len(nrfi_rows),
            "brier": float(np.mean((p_nrfi - nrfi_act) ** 2)),
            "pred_rate": float(p_nrfi.mean()), "actual_rate": float(nrfi_act.mean()),
            "bias": float(p_nrfi.mean() - nrfi_act.mean()),
            "ece": ece_n, "reliability": rel_n,
        }

    # ---- starter strikeouts (home + away pooled) ----
    kp, ka, pov, kov = [], [], [], []
    for r in rows:
        for side in ("home", "away"):
            act = r[f"{side}_sp_k_actual"]
            if act is None:
                continue
            kp.append(r[f"{side}_sp_k_pred"]); ka.append(act)
            pov.append(r[f"p_{side}_sp_over"]); kov.append(int(act > K_LINE))
    if kp:
        kp, ka, pov, kov = map(np.array, (kp, ka, pov, kov))
        rel_k, ece_k = reliability(pov, kov.astype(float))
        out["starter_k"] = {
            "n": len(kp),
            "mae": float(np.mean(np.abs(kp - ka))),
            "pred_mean": float(kp.mean()), "actual_mean": float(ka.mean()),
            "bias": float(kp.mean() - ka.mean()),
            "over_line": K_LINE, "pred_over_rate": float(pov.mean()),
            "actual_over_rate": float(kov.mean()),
            "ece": ece_k, "reliability": rel_k,
        }
    return out


def _verdict(bias, ece, bias_tol, ece_tol=0.07):
    ok = abs(bias) <= bias_tol and ece <= ece_tol
    return "READY" if ok else "NEEDS WORK"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--rate-season", type=int, default=dt.date.today().year - 1)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    r = run(args.season, args.rate_season, args.sims, args.limit)
    if r.get("n", 0) == 0:
        print("no games to evaluate yet"); raise SystemExit

    print(f"\nProps/totals on trial over {r['n']} games:\n")

    t = r["totals"]
    print(f"TOTAL RUNS  pred {t['pred_mean']:.2f} vs actual {t['actual_mean']:.2f} "
          f"(bias {t['bias']:+.2f}, MAE {t['mae']:.2f})")
    print(f"  over {t['over_line']}: sim says {t['pred_over_rate']:.1%}, "
          f"actual {t['actual_over_rate']:.1%} | calibration err {t['ece']:.3f}")
    print(f"  reliability {_fmt_reliability(t['reliability'])}")
    print(f"  => {_verdict(t['bias'], t['ece'], bias_tol=0.4)}\n")

    if "nrfi" in r:
        n = r["nrfi"]
        print(f"NRFI  sim says {n['pred_rate']:.1%}, actual {n['actual_rate']:.1%} "
              f"(bias {n['bias']:+.3f}, Brier {n['brier']:.3f})")
        print(f"  reliability {_fmt_reliability(n['reliability'])}")
        print(f"  => {_verdict(n['bias'], n['ece'], bias_tol=0.05)}\n")

    if "starter_k" in r:
        k = r["starter_k"]
        print(f"STARTER K (n={k['n']})  pred {k['pred_mean']:.2f} vs actual "
              f"{k['actual_mean']:.2f} (bias {k['bias']:+.2f}, MAE {k['mae']:.2f})")
        print(f"  over {k['over_line']}: sim says {k['pred_over_rate']:.1%}, "
              f"actual {k['actual_over_rate']:.1%} | calibration err {k['ece']:.3f}")
        print(f"  reliability {_fmt_reliability(k['reliability'])}")
        print(f"  => {_verdict(k['bias'], k['ece'], bias_tol=0.5)}")
