"""
run_slate.py
------------
Predict and log EVERY game on a date in one shot, so the dashboard covers the
whole MLB slate without running games one at a time.

  python run_slate.py                       # today, 2025 rates, model+market logged
  python run_slate.py --date 2026-06-25
  python run_slate.py --blend               # use 2026-to-date blended onto 2025 (pull 2026 statcast first)
  python run_slate.py --dry-run             # just list what it WOULD log

Discipline kept intact:
  - logs the MODEL number first, then attaches the de-vigged market price
  - only logs games that haven't started yet (pre-first-pitch), so every logged
    number is honest out-of-sample
  - idempotent: a game already logged for that date is skipped, so you can run it
    repeatedly through the morning as more probables/lineups get confirmed

After it runs:
  python export_dashboard.py --season 2026
  git add dashboard/ledger.json && git commit -m "slate" && git push
"""
from __future__ import annotations

import argparse
import datetime as dt

import config
from ingest.pull_mlb_statsapi import schedule
from features.load_teams import build_game, load_rate_tables, load_blended_rate_tables
from run_predict import card
from ledger import ledger

TEAM_NAMES = {
    "ARI": "Diamondbacks", "AZ": "Diamondbacks", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CWS": "White Sox", "CIN": "Reds", "CLE": "Guardians",
    "COL": "Rockies", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
    "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
    "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "ATH": "Athletics",
    "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SEA": "Mariners",
    "SF": "Giants", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WSH": "Nationals",
}

# game has not started yet -> safe to log a pre-first-pitch number
STARTED = ("In Progress", "Final", "Game Over", "Completed", "Postponed",
           "Suspended", "Cancelled", "Delayed")


def _already_logged(rows, game_id):
    return any(r["game_id"] == game_id and r["market"] == "moneyline_home" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--season", type=int, default=2025, help="prior-season rate snapshots")
    ap.add_argument("--blend", action="store_true",
                    help="blend current-season-to-date onto prior season (needs 2026 statcast)")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--all", action="store_true", help="log even started/final games")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    date = "today" if args.date == "today" else args.date
    cur_year = dt.date.today().year if date == "today" else int(date[:4])

    games = schedule(date)
    if not games:
        print(f"[slate] no games found for {date}")
        return
    print(f"[slate] {len(games)} games scheduled for {date}")

    # load rate tables ONCE
    tables = None
    if not args.dry_run:
        try:
            tables = (load_blended_rate_tables(cur_year, args.season) if args.blend
                      else load_rate_tables(args.season))
            print(f"[slate] rates: {'blended ' + str(cur_year) + '+' if args.blend else ''}{args.season}")
        except FileNotFoundError as ex:
            print(f"[slate] rate snapshots missing: {ex}")
            return

    # market snapshot (optional)
    winner_snap = None
    try:
        from market.kalshi_match import load_snapshot
        winner_snap = load_snapshot(date, "winner")
    except Exception:
        pass

    rows = ledger._load()
    logged = skipped = 0
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
            h_t, a_t, ctx, info = build_game(spec, real_date, args.season, tables=tables)
            res = card(h_t, a_t, ctx, n_sims=args.sims)
        except Exception as ex:
            print(f"  [skip] {spec}: {ex}")
            continue

        # market price for the moneyline, if we have a snapshot
        mkt_home_p = None
        if winner_snap:
            try:
                from market.kalshi_match import winner_edge
                w = winner_edge(winner_snap, away, home, res["p_home_win"])
                if w and "market_home_p" in w:
                    mkt_home_p = w["market_home_p"]
            except Exception:
                pass

        meta = {"home": home, "away": away, "date": real_date,
                "home_name": TEAM_NAMES.get(home, home),
                "away_name": TEAM_NAMES.get(away, away),
                "park": ctx.park_code, "lineup_source": info.get("lineup_source")}
        ledger.log_prediction(gid, "moneyline_home", res["p_home_win"],
                              market_p=mkt_home_p, meta=meta)
        ledger.log_prediction(gid, "total_over_8.5", res["total_over"]["over_8.5"], meta=meta)
        ledger.log_prediction(gid, "nrfi", res["p_nrfi"], meta=meta)
        rows = ledger._load()      # refresh so idempotency holds within the run
        logged += 1
        mk = f"{mkt_home_p:.3f}" if mkt_home_p is not None else "n/a"
        print(f"  {spec:9s} model {res['p_home_win']:.3f}  market {mk}")

    verb = "would log" if args.dry_run else "logged"
    print(f"\n[slate] {verb} {logged} games, skipped {skipped} "
          f"(already logged / started)")
    if not args.dry_run and logged:
        print("next: python export_dashboard.py --season {0} && git add dashboard/ledger.json "
              "&& git commit -m slate && git push".format(cur_year))


if __name__ == "__main__":
    main()
