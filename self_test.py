"""
self_test.py
------------
Runs entirely on your LOCAL data (no API). Two checks:

1. Sanity-check config.LEAGUE_PA_RATES (order + sums to 1).
2. Build a deliberately GREAT team (top HR hitters + top-K starter) vs a
   deliberately BAD team (worst hitters + most-hittable starter) from your
   2025 rate tables, and simulate. The great team should win ~75-80%.

If the great team LOSES (<0.5), the inversion is in your local rates/engine.
If it WINS, your core is fine and the bug is in the per-game reconstruction.

  python self_test.py --rate-season 2025
"""
from __future__ import annotations

import argparse
import datetime as dt

import config
from features.load_teams import load_rate_tables, build_team
from sim.markov_game import GameContext, run_simulation


def main(rate_season: int):
    print("=== config check ===")
    print("EVENTS order:", config.EVENTS)
    s = sum(config.LEAGUE_PA_RATES.values())
    print("LEAGUE_PA_RATES sum:", round(s, 5), "(should be 1.0)")
    print("  HR:", config.LEAGUE_PA_RATES["HR"], " K:", config.LEAGUE_PA_RATES["K"],
          " IP_OUT:", config.LEAGUE_PA_RATES["IP_OUT"])

    print("\n=== loading your rate tables ===")
    tables = load_rate_tables(rate_season)
    bat, pit = tables["bat"], tables["pit"]
    print(f"batters {len(bat)}, pitchers {len(pit)}")
    print(f"handedness cache: {len(tables['bhand'])} batters, {len(tables['phand'])} pitchers")

    # GREAT lineup: highest HR rate among regulars; BAD: highest (K+IP_OUT)
    clean = bat.dropna(subset=["HR", "K", "IP_OUT"]).copy()
    thresh = 200
    while len(clean[clean["PA"] >= thresh]) < 20 and thresh > 25:
        thresh -= 25
    reg = clean[clean["PA"] >= thresh].copy()
    print(f"(using PA>={thresh}, {len(reg)} qualifying batters)")
    great_bats = reg.sort_values("HR", ascending=False).head(9)
    reg["weak"] = reg["K"] + reg["IP_OUT"]
    bad_bats = reg.sort_values("weak", ascending=False).head(9)
    pit_clean = pit.dropna(subset=["K", "HR"])
    pit_reg = pit_clean[pit_clean["PA"] >= thresh]
    great_sp = int(pit_reg.sort_values("K", ascending=False).index[0])   # ace
    bad_sp = int(pit_reg.sort_values("HR", ascending=False).index[0])    # gets crushed

    great_line = [(int(i), f"G{n+1}") for n, i in enumerate(great_bats.index)]
    bad_line = [(int(i), f"B{n+1}") for n, i in enumerate(bad_bats.index)]

    print(f"\nGREAT lineup mean HR rate: {great_bats['HR'].mean():.3f}")
    print(f"BAD   lineup mean HR rate: {bad_bats['HR'].mean():.3f}")

    great = build_team("GREAT", great_line, great_sp, "Ace", tables)
    bad = build_team("BAD", bad_line, bad_sp, "BP", tables)

    # GREAT at home vs BAD away -> GREAT should dominate
    res = run_simulation(great, bad, GameContext(), n_sims=6000, seed=1)
    p = res["p_home_win"]
    print(f"\nGREAT (home) P(win): {p:.3f}   expected total runs: {res['exp_total']:.2f}")
    print("=>", "CORRECT - core is healthy" if p > 0.6
          else ("INVERTED - bug is in local rates/engine" if p < 0.5
                else "WEAK - something is dampening team strength"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-season", type=int, default=dt.date.today().year - 1)
    main(ap.parse_args().rate_season)
