"""
export_dashboard.py
-------------------
Turn the prediction ledger into the ledger.json the public dashboard reads.
v1 exports the validated WINNER market only (model vs market, per game). Run it
after predictions/results update, then commit dashboard/ledger.json.

  python export_dashboard.py --season 2026 --out dashboard/ledger.json

It auto-resolves finals by cross-referencing results_<season>.json (by date +
teams), so there's no manual resolve step. Output matches the dashboard fields:
  date, away, home, away_name, home_name,
  model_home_win_pct, market_home_win_pct, edge, contrarian, logged_at,
  status ("upcoming"|"final"),
  finals also: away_score, home_score, winner, pick_correct
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import config

WINNER_MARKET = "moneyline_home"   # market string logged by run_predict
CONTRARIAN_EDGE = 8.0              # |model - market| pct points flagging a contrarian call


def _parse_gid(gid: str, meta: dict):
    """game_id is 'AWAY@HOME-YYYY-MM-DD'. meta may also carry the fields."""
    away = meta.get("away"); home = meta.get("home"); date = meta.get("date")
    if (not away or not home or not date) and "@" in gid:
        away_part, _, rest = gid.partition("@")     # 'NYM', 'PHI-2026-06-23'
        home_part, _, date_part = rest.partition("-")
        away = away or away_part
        home = home or home_part
        date = date or date_part
    return away or "?", home or "?", date or ""


def export(season: int) -> list[dict]:
    ledger = json.loads(config.LEDGER_PATH.read_text()) if config.LEDGER_PATH.exists() else []
    rpath = config.SNAPSHOTS / f"results_{season}.json"
    res_list = json.loads(rpath.read_text()) if rpath.exists() else []
    res_idx = {(g["date"], g["home"], g["away"]): g for g in res_list}

    out = []
    for r in ledger:
        if r.get("market") != WINNER_MARKET:
            continue
        model_p = r.get("model_p")
        if model_p is None:
            continue
        meta = r.get("meta", {})
        away, home, date = _parse_gid(str(r["game_id"]), meta)
        market_p = r.get("market_p")
        model_pct = round(model_p * 100, 1)
        market_pct = None if market_p is None else round(market_p * 100, 1)
        edge = None if market_pct is None else round(model_pct - market_pct, 1)

        entry = {
            "date": date, "away": away, "home": home,
            "away_name": meta.get("away_name", away),
            "home_name": meta.get("home_name", home),
            "model_home_win_pct": model_pct,
            "market_home_win_pct": market_pct,
            "edge": edge,
            "contrarian": bool(edge is not None and abs(edge) >= CONTRARIAN_EDGE),
            "logged_at": r.get("ts", ""),
            "status": "upcoming",
        }

        g = res_idx.get((date, home, away))
        if g and g.get("home_score") is not None:
            hs, as_ = g["home_score"], g["away_score"]
            entry["status"] = "final"
            entry["home_score"] = hs
            entry["away_score"] = as_
            entry["winner"] = home if hs > as_ else away
            entry["pick_correct"] = bool((model_p > 0.5) == (hs > as_))
        out.append(entry)

    out.sort(key=lambda e: (e["date"], e["logged_at"]), reverse=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--out", default="dashboard/ledger.json")
    args = ap.parse_args()
    data = export(args.season)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2))
    finals = [d for d in data if d["status"] == "final"]
    correct = sum(d.get("pick_correct", False) for d in finals)
    print(f"wrote {len(data)} games -> {outp}")
    if finals:
        print(f"  {len(finals)} final, picks correct {correct}/{len(finals)} ({correct/len(finals):.1%})")
