"""
run_predict.py
--------------
End-to-end single-game prediction. In production this wires together:
  schedule/probables (statsapi) -> PA rates (statcast) -> sim -> de-vig Kalshi -> ledger

For now it ships with a self-contained DEMO matchup so you can run the whole
flow today, before any data pull. Replace `demo_teams()` with real loaders as
the ingest layer fills in.

  python run_predict.py            # runs the demo matchup and prints a card
  python run_predict.py --log      # also writes predictions to the ledger
"""
from __future__ import annotations

import argparse
import json

from features.pa_probabilities import batter_from_slash, pitcher_from_rates
from sim.markov_game import Team, GameContext, run_simulation
from market.devig import fair_two_way, edge
from ledger import ledger


def demo_teams():
    """A representative matchup: a strong-K road ace vs a power home lineup."""
    # Home: above-average power lineup (high HR%, average K)
    home_lineup = [
        batter_from_slash("Tovar",    k_pct=0.20, bb_pct=0.06, hr_pct=0.030, hand="R", order=1),
        batter_from_slash("Doyle",    k_pct=0.27, bb_pct=0.08, hr_pct=0.045, hand="R", order=2),
        batter_from_slash("Blackmon", k_pct=0.18, bb_pct=0.09, hr_pct=0.035, hand="L", order=3),
        batter_from_slash("McMahon",  k_pct=0.28, bb_pct=0.10, hr_pct=0.045, hand="L", order=4),
        batter_from_slash("Diaz",     k_pct=0.17, bb_pct=0.07, hr_pct=0.040, hand="R", order=5),
        batter_from_slash("Montero",  k_pct=0.24, bb_pct=0.07, hr_pct=0.038, hand="R", order=6),
        batter_from_slash("Rodgers",  k_pct=0.20, bb_pct=0.06, hr_pct=0.028, hand="R", order=7),
        batter_from_slash("Cave",     k_pct=0.23, bb_pct=0.07, hr_pct=0.030, hand="L", order=8),
        batter_from_slash("Stallings",k_pct=0.26, bb_pct=0.08, hr_pct=0.025, hand="R", order=9),
    ]
    # Away: league-average sticks
    away_lineup = [
        batter_from_slash(f"A{i+1}", k_pct=0.225, bb_pct=0.085, hr_pct=0.032,
                          hand="RL"[i % 2], order=i + 1) for i in range(9)
    ]
    home = Team("COL", home_lineup,
                pitcher_from_rates("Freeland", k_pct=0.18, bb_pct=0.08, hr_pct=0.038, hand="L"),
                pitcher_from_rates("COL_pen", k_pct=0.21, bb_pct=0.09, hr_pct=0.036))
    away = Team("LAD", away_lineup,
                pitcher_from_rates("Ace", k_pct=0.30, bb_pct=0.055, hr_pct=0.025, hand="R"),
                pitcher_from_rates("LAD_pen", k_pct=0.26, bb_pct=0.07, hr_pct=0.030))
    return home, away


def card(home, away, ctx, n_sims=20000):
    res = run_simulation(home, away, ctx, n_sims=n_sims, seed=7)
    print(f"\n=== {away.code} @ {home.code}  ({ctx.park_code} park, {n_sims:,} sims) ===")
    print(f"Moneyline   home {res['p_home_win']:.3f}  away {res['p_away_win']:.3f}")
    print(f"Total       E[runs] {res['exp_total']:.2f}   "
          f"over 8.5 {res['total_over']['over_8.5']:.3f}")
    print(f"Run line    home -1.5 {res['p_home_-1.5']:.3f}")
    print(f"First inn   NRFI {res['p_nrfi']:.3f}   YRFI {res['p_yrfi']:.3f}")
    print(f"F5          home {res['p_f5_home']:.3f}   first-to-score home {res['p_first_to_score_home']:.3f}")
    print(f"{away.starter.name} (away SP) Ks  mean {res['away_starter_k']['mean']:.2f}  "
          f"over 6.5 {res['away_starter_k']['over']['over_6.5']:.3f}")
    print("Top home HR threats:")
    for b in sorted(res["home_batters"], key=lambda x: -x["p_hr"])[:3]:
        print(f"   {b['name']:<10} P(HR) {b['p_hr']:.3f}  E[TB] {b['exp_tb']:.2f}")
    return res


def example_market_compare(model_home_p: float):
    """Show the de-vig + edge step with an illustrative Kalshi pair."""
    # e.g. Kalshi shows home YES ask 56c, NO ask 47c (overround 1.03)
    fair = fair_two_way(56, 47, method="power")
    e = edge(model_home_p, fair["fair_yes"])
    print(f"\nMarket (illustrative): home fair {fair['fair_yes']:.3f} "
          f"(overround {fair['overround']:.3f}, {fair['method']})")
    print(f"Model edge on home ML: {e:+.3f}")
    return fair, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", help="real matchup, e.g. 'LAD@COL' (away@home)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (defaults to today)")
    ap.add_argument("--season", type=int, default=2025,
                    help="season whose rate snapshots to use")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    if args.game:
        run_real_game(args.game, args.date, args.season, args.sims, args.log)
        return

    # ---- demo path (no data needed) ----
    home, away = demo_teams()
    ctx = GameContext(park_code="COL", ump_k_mult=1.0)   # Coors
    res = card(home, away, ctx, n_sims=args.sims)
    fair, e = example_market_compare(res["p_home_win"])

    if args.log:
        gid = f"{away.code}@{home.code}-demo"
        ledger.log_prediction(gid, "moneyline_home", res["p_home_win"],
                              market_p=fair["fair_yes"],
                              meta={"park": ctx.park_code})
        ledger.log_prediction(gid, "total_over_8.5",
                              res["total_over"]["over_8.5"])
        ledger.log_prediction(gid, "nrfi", res["p_nrfi"])
        print("\n[ledger] logged 3 predictions ->", ledger.LEDGER.name)


def run_real_game(spec, date, season, n_sims, do_log):
    """Build a real matchup from snapshots and print + optionally log its card."""
    import datetime as dt
    from features.load_teams import build_game
    date = date or dt.date.today().isoformat()
    try:
        home, away, ctx, info = build_game(spec, date, season)
    except (FileNotFoundError, ValueError) as ex:
        print(f"[run] {ex}")
        return

    src = info["lineup_source"]
    print(f"\nVenue: {info['venue']}   Probables: "
          f"{away.code} {info['away_sp']} vs {home.code} {info['home_sp']}")
    print(f"Lineups -> home: {src['home'].upper()}, away: {src['away'].upper()}")
    if "projected" in src.values() or "fallback" in src.values():
        print("  (!) lineup not yet confirmed - order is a PROXY. Re-run after "
              "official lineups post ~1-3 hrs before first pitch.")

    res = card(home, away, ctx, n_sims=n_sims)

    # baseline floor (Elo + Pythag) if results snapshot exists
    # rates use last completed season; baselines use the CURRENT season's results
    results_season = int(date[:4])
    try:
        from models.baselines import predict as baseline_predict
        bp = baseline_predict(home.code, away.code, results_season)
        print("\n--- baselines (floor the sim must beat) ---")
        elo_p = bp.get("elo_home_p")
        py_p = bp.get("pythag_home_p")
        print(f"Elo     P(home) {elo_p:.3f}   (ratings {bp.get('elo_home_rating')} / {bp.get('elo_away_rating')})")
        if py_p is not None:
            print(f"Pythag  P(home) {py_p:.3f}   (win% {bp.get('home_winpct')} / {bp.get('away_winpct')})")
        print(f"Sim     P(home) {res['p_home_win']:.3f}")
    except FileNotFoundError:
        pass  # no results snapshot yet; skip silently

    # model vs market (Kalshi) edge lines
    from market.kalshi_match import print_edges, load_snapshot, winner_edge
    away_code, home_code = spec.replace("@", " ").upper().split()
    print_edges(date, away_code, home_code, res)

    if do_log:
        gid = f"{spec}-{date}"
        # attach de-vigged market price to the moneyline row if we have it
        mkt_home_p = None
        win = load_snapshot(date, "winner")
        if win:
            w = winner_edge(win, away_code, home_code, res["p_home_win"])
            if w and "market_home_p" in w:
                mkt_home_p = w["market_home_p"]
        ledger.log_prediction(gid, "moneyline_home", res["p_home_win"],
                              market_p=mkt_home_p,
                              meta={"park": ctx.park_code, "lineup_source": src})
        ledger.log_prediction(gid, "total_over_8.5", res["total_over"]["over_8.5"])
        ledger.log_prediction(gid, "nrfi", res["p_nrfi"])
        print(f"\n[ledger] logged 3 predictions for {gid}")


if __name__ == "__main__":
    main()
