"""
run_slate.py
------------
Predict and log EVERY game on a date in one shot, so the dashboard covers the
whole MLB slate without running games one at a time.

  python run_slate.py                       # today: pull market, log model-vs-market
  python run_slate.py --confirmed-only      # only log games whose real lineup is posted
  python run_slate.py --no-market           # skip the Kalshi pull (model-only)
  python run_slate.py --blend               # 2026-to-date blended onto 2025 (pull 2026 statcast first)
  python run_slate.py --dry-run             # list what it WOULD log

Discipline kept intact:
  - logs the MODEL number first, then attaches the de-vigged market price
  - only logs games that haven't started yet (pre-first-pitch)
  - idempotent: a game already logged for that date is skipped, so you can run it
    repeatedly through the afternoon as lineups confirm and prices move

After it runs:
  python export_dashboard.py --season 2026
  git add dashboard/ledger.json && git commit -m slate && git push
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import config
from ingest.pull_mlb_statsapi import snapshot_probables, _date
from features.load_teams import build_game, load_rate_tables, load_blended_rate_tables
from run_predict import card
from ledger import ledger

TEAM_NAMES = {
    "ARI": "Diamondbacks", "AZ": "Diamondbacks", "ATL": "Braves", "BAL": "Orioles",
    "BOS": "Red Sox", "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds",
    "CLE": "Guardians", "COL": "Rockies", "DET": "Tigers", "HOU": "Astros",
    "KC": "Royals", "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins",
    "MIL": "Brewers", "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees",
    "ATH": "Athletics", "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres",
    "SEA": "Mariners", "SF": "Giants", "STL": "Cardinals", "TB": "Rays",
    "TEX": "Rangers", "TOR": "Blue Jays", "WSH": "Nationals",
}

STARTED = ("In Progress", "Final", "Game Over", "Completed", "Postponed",
           "Suspended", "Cancelled", "Delayed")


def _already_logged(rows, game_id):
    return any(r["game_id"] == game_id and r["market"] == "moneyline_home" for r in rows)


def _pull_market(no_market: bool):
    if no_market:
        return None
    try:
        from ingest.pull_kalshi import snapshot as kalshi_snapshot
        path = kalshi_snapshot("winner")
        markets = json.loads(path.read_text()).get("markets", [])
        print(f"[slate] market: {len(markets)} Kalshi winner markets")
        return markets
    except Exception as ex:
        print(f"[slate] Kalshi pull failed ({type(ex).__name__}); logging model-only")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--season", type=int, default=2025, help="prior-season rate snapshots")
    ap.add_argument("--blend", action="store_true",
                    help="blend current-season-to-date onto prior season (needs 2026 statcast)")
    ap.add_argument("--confirmed-only", action="store_true",
                    help="only log a game once BOTH real lineups are posted")
    ap.add_argument("--no-market", action="store_true", help="skip the Kalshi pull")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--all", action="store_true", help="log even started/final games")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snap_date = _date(args.date)
    cur_year = int(snap_date[:4])
    games = snapshot_probables(snap_date)        # fetch live schedule + WRITE snapshot
    if not games:
        print(f"[slate] no games found for {snap_date}")
        return

    tables = None
    winner_snap = None
    if not args.dry_run:
        try:
            tables = (load_blended_rate_tables(cur_year, args.season,
                                               xrates=args.xrates) if args.blend
                      else load_rate_tables(args.season, xrates=args.xrates))
            print(f"[slate] rates: {'blended ' + str(cur_year) + '+' if args.blend else ''}"
                  f"{args.season}{' (xrates)' if args.xrates else ''}"
                  f"{' +env' if args.env else ''}{' +workload' if args.workload else ''}"
                  f"{' +availability' if args.availability else ''}")
        except FileNotFoundError as ex:
            print(f"[slate] rate snapshots missing: {ex}")
            return
        winner_snap = _pull_market(args.no_market)

    rows = ledger._load()
    logged = skipped = 0
    not_ready = []
    for g in games:
        home, away = g["home"], g["away"]
        spec = f"{away}@{home}"
        real_date = g["gameDate"][:10]
        gid = f"{spec}-{real_date}"

        if not args.all and any(s in g["status"] for s in STARTED):
            skipped += 1; continue
        if _already_logged(rows, gid):
            skipped += 1; continue

        if args.dry_run:
            print(f"  would log {spec:9s} {g['status']:12s} "
                  f"SP {g.get('away_probable') or 'TBD'} @ {g.get('home_probable') or 'TBD'}")
            logged += 1
            continue

        try:
            h_t, a_t, ctx, info = build_game(spec, snap_date, args.season, tables=tables,
                                             env=args.env, workload=args.workload,
                                             availability=args.availability)
        except Exception as ex:
            print(f"  [skip] {spec}: {ex}")
            continue

        src = info.get("lineup_source", {})
        confirmed = src.get("home") == "confirmed" and src.get("away") == "confirmed"
        if args.confirmed_only and not confirmed:
            not_ready.append(spec)
            continue

        res = card(h_t, a_t, ctx, n_sims=args.sims)

        mkt_home_p = None
        if winner_snap:
            try:
                from market.kalshi_match import winner_edge
                w = winner_edge(winner_snap, away, home, res["p_home_win"])
                if w and w.get("market_home_p") is not None:
                    mkt_home_p = w["market_home_p"]
            except Exception:
                pass

        meta = {"home": home, "away": away, "date": real_date,
                "home_name": TEAM_NAMES.get(home, home),
                "away_name": TEAM_NAMES.get(away, away),
                "park": ctx.park_code,
                "lineup_source": "confirmed" if confirmed else "projected"}
        ledger.log_prediction(gid, "moneyline_home", res["p_home_win"],
                              market_p=mkt_home_p, meta=meta)
        ledger.log_prediction(gid, "total_over_8.5", res["total_over"]["over_8.5"], meta=meta)
        ledger.log_prediction(gid, "nrfi", res["p_nrfi"], meta=meta)
        ledger.log_prediction(gid, "away_sp_k_over_5.5",
                              res["away_starter_k"]["over"]["over_5.5"],
                              meta={**meta, "sp": g.get("away_probable") or "TBD"})
        ledger.log_prediction(gid, "home_sp_k_over_5.5",
                              res["home_starter_k"]["over"]["over_5.5"],
                              meta={**meta, "sp": g.get("home_probable") or "TBD"})
        # validated per-batter HR props: log the top threats across both lineups
        threats = ([{"name": b["name"], "p_hr": round(b["p_hr"], 3), "team": home}
                    for b in res["home_batters"]]
                   + [{"name": b["name"], "p_hr": round(b["p_hr"], 3), "team": away}
                      for b in res["away_batters"]])
        threats.sort(key=lambda x: -x["p_hr"])
        top = threats[:3]
        ledger.log_prediction(gid, "batter_hr", top[0]["p_hr"],
                              meta={**meta, "hr_threats": top})
        rows = ledger._load()
        logged += 1
        mk = f"{mkt_home_p:.3f}" if mkt_home_p is not None else "n/a"
        tag = "" if confirmed else " (projected)"
        print(f"  {spec:9s} model {res['p_home_win']:.3f}  market {mk}{tag}")

    verb = "would log" if args.dry_run else "logged"
    print(f"\n[slate] {verb} {logged} games, skipped {skipped} (already logged / started)")
    if not_ready:
        print(f"[slate] {len(not_ready)} waiting on lineups: {', '.join(not_ready)}")
        print("        re-run later; confirmed games will log then (already-logged ones are skipped)")
    if not args.dry_run and logged:
        print(f"next: python export_dashboard.py --season {cur_year} "
              "&& git add dashboard/ledger.json && git commit -m slate && git push")


if __name__ == "__main__":
    main()