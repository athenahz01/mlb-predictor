"""
export_dashboard.py  (v2)
-------------------------
Turn the prediction ledger into the JSON the public dashboard reads.

Output shape:
  {
    "generated_at": iso,
    "summary": { honest model-vs-market calibration over games with a REAL market price },
    "games":   [ per-game winner + props ]
  }

Run after predictions/results update, then commit dashboard/ledger.json.
  python export_dashboard.py --season 2026 --out dashboard/ledger.json

Auto-resolves finals by cross-referencing results_<season>.json (date + teams).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import config

WINNER_MARKET = "moneyline_home"
CONTRARIAN_EDGE = 8.0


def _parse_gid(gid: str, meta: dict):
    away = meta.get("away"); home = meta.get("home"); date = meta.get("date")
    if (not away or not home or not date) and "@" in gid:
        away_part, _, rest = gid.partition("@")
        home_part, _, date_part = rest.partition("-")
        away = away or away_part; home = home or home_part; date = date or date_part
    return away or "?", home or "?", date or ""


def export(season: int) -> dict:
    ledger = json.loads(config.LEDGER_PATH.read_text()) if config.LEDGER_PATH.exists() else []
    rpath = config.SNAPSHOTS / f"results_{season}.json"
    res_list = json.loads(rpath.read_text()) if rpath.exists() else []
    res_idx = {(g["date"], g["home"], g["away"]): g for g in res_list}

    # group every market row by game id
    by_game = defaultdict(dict)
    for r in ledger:
        by_game[str(r["game_id"])][r.get("market")] = r

    games = []
    for gid, markets in by_game.items():
        win = markets.get(WINNER_MARKET)
        if not win or win.get("model_p") is None:
            continue
        meta = win.get("meta", {})
        away, home, date = _parse_gid(gid, meta)
        model_p = win["model_p"]; market_p = win.get("market_p")
        model_pct = round(model_p * 100, 1)
        market_pct = None if market_p is None else round(market_p * 100, 1)
        edge = None if market_pct is None else round(model_pct - market_pct, 1)

        def prob(mkt):
            r = markets.get(mkt)
            return None if not r or r.get("model_p") is None else round(r["model_p"] * 100, 1)

        entry = {
            "date": date, "away": away, "home": home,
            "away_name": meta.get("away_name", away),
            "home_name": meta.get("home_name", home),
            "model_home_win_pct": model_pct,
            "market_home_win_pct": market_pct,
            "edge": edge,
            "contrarian": bool(edge is not None and abs(edge) >= CONTRARIAN_EDGE),
            "logged_at": win.get("ts", ""),
            "lineup_source": meta.get("lineup_source"),
            "status": "upcoming",
            # props (model probabilities; None until that market is logged)
            "props": {
                "nrfi_pct": prob("nrfi"),
                "over_8_5_pct": prob("total_over_8.5"),
                "away_sp_k_over_5_5_pct": prob("away_sp_k_over_5.5"),
                "home_sp_k_over_5_5_pct": prob("home_sp_k_over_5.5"),
                "away_sp": (markets.get("away_sp_k_over_5.5", {}).get("meta", {}) or {}).get("sp"),
                "home_sp": (markets.get("home_sp_k_over_5.5", {}).get("meta", {}) or {}).get("sp"),
                "hr_threats": [
                    {"name": t["name"], "team": t["team"], "pct": round(t["p_hr"] * 100, 1)}
                    for t in ((markets.get("batter_hr", {}).get("meta", {}) or {}).get("hr_threats") or [])
                ] or None,
            },
        }

        g = res_idx.get((date, home, away))
        if g and g.get("home_score") is not None:
            hs, as_ = g["home_score"], g["away_score"]
            entry["status"] = "final"
            entry["home_score"] = hs; entry["away_score"] = as_
            entry["winner"] = home if hs > as_ else away
            entry["pick_correct"] = bool((model_p > 0.5) == (hs > as_))
            entry["total_runs"] = hs + as_
        games.append(entry)

    games.sort(key=lambda e: (e["date"], e["logged_at"]), reverse=True)

    # honest summary: calibration ONLY over finals that have a real market price
    finals = [g for g in games if g["status"] == "final"]
    compared = [g for g in finals if g["market_home_win_pct"] is not None]
    def brier(g, which):
        outcome = 1.0 if g["winner"] == g["home"] else 0.0
        p = g[which] / 100.0
        return (p - outcome) ** 2
    model_brier = round(sum(brier(g, "model_home_win_pct") for g in compared) / len(compared), 3) if compared else None
    market_brier = round(sum(brier(g, "market_home_win_pct") for g in compared) / len(compared), 3) if compared else None
    correct = sum(1 for g in finals if g.get("pick_correct"))
    summary = {
        "n_finals": len(finals),
        "n_compared": len(compared),          # games with both model and market
        "model_brier": model_brier,
        "market_brier": market_brier,
        "picks_correct": correct,
        "picks_total": len(finals),
        "accuracy": round(correct / len(finals), 3) if finals else None,
    }
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "summary": summary, "games": games}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--out", default="dashboard/ledger.json")
    args = ap.parse_args()
    data = export(args.season)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    # ledger.json stays a plain ARRAY (backward-compatible; props added per game)
    outp.write_text(json.dumps(data["games"], indent=2))
    # honest calibration headline written alongside as summary.json
    (outp.parent / "summary.json").write_text(json.dumps(data["summary"], indent=2))
    s = data["summary"]
    print(f"wrote {len(data['games'])} games -> {outp}")
    print(f"wrote calibration headline -> {outp.parent / 'summary.json'}")
    print(f"  finals {s['n_finals']}, picks {s['picks_correct']}/{s['picks_total']} "
          f"({(s['accuracy'] or 0):.1%})")
    if s["n_compared"]:
        print(f"  calibration over {s['n_compared']} games w/ market: "
              f"model Brier {s['model_brier']} vs market {s['market_brier']}")
    else:
        print("  no games with real market price yet (headline stays empty until there are)")