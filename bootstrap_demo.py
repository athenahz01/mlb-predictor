"""
bootstrap_demo.py
-----------------
A FILMABLE version of the paired bootstrap. It does the same luck-check the
backtest does, but out loud: it reshuffles the games thousands of times, ticks
a counter as it goes, draws a little text histogram of the results, and lands
on the p-value and verdict.

Reads your existing sim backtest cache + results, so run the backtest first.

  python bootstrap_demo.py --season 2026 --rate-season 2025 --iters 10000
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import numpy as np

import config
from backtest.sim_backtest import walk_forward_baselines
from backtest.walk_forward import logloss


def main(season: int, rate_season: int, iters: int):
    sim = json.loads((config.SNAPSHOTS / f"sim_backtest_{season}.json").read_text())
    results = json.loads((config.SNAPSHOTS / f"results_{season}.json").read_text())
    results.sort(key=lambda g: (g["date"], g["gamePk"]))
    base = walk_forward_baselines(results)

    rows = [(sim[str(g["gamePk"])], base[g["gamePk"]]["elo_p"],
             base[g["gamePk"]]["y"]) for g in results
            if str(g["gamePk"]) in sim and g["gamePk"] in base]
    sp = np.array([r[0] for r in rows]); ep = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows], float)

    L_sim = logloss(sp, y); L_elo = logloss(ep, y)
    real_diff = L_elo.mean() - L_sim.mean()   # positive = sim better

    print(f"\n{len(rows)} games. real edge of sim over Elo: {real_diff:+.4f} log-loss")
    print("(higher = sim better. now: is this real, or luck?)\n")
    print("reshuffling the games to see how often pure luck beats this edge...\n")
    time.sleep(0.8)

    n = len(rows)
    diffs = np.empty(iters)
    beats = 0
    rng = np.random.default_rng(0)
    bar_len = 30
    for i in range(iters):
        idx = rng.integers(0, n, n)               # resample games with replacement
        d = L_elo[idx].mean() - L_sim[idx].mean()
        diffs[i] = d
        if d <= 0:
            beats += 1
        if (i + 1) % max(1, iters // 200) == 0 or i + 1 == iters:
            done = (i + 1) / iters
            fill = int(bar_len * done)
            print(f"\r  reshuffle {i+1:>6}/{iters}  [{'#'*fill}{'.'*(bar_len-fill)}]  "
                  f"luck beat it {beats} times", end="", flush=True)
    print("\n")

    # tiny text histogram
    lo, hi = diffs.min(), diffs.max()
    bins = 31
    counts, edges = np.histogram(diffs, bins=bins, range=(lo, hi))
    peak = counts.max()
    zero_bin = int((0 - lo) / (hi - lo) * (bins - 1)) if hi > lo else 0
    print("  distribution of the edge across reshuffles (| marks zero):\n")
    for b in range(bins):
        h = int(28 * counts[b] / peak) if peak else 0
        mark = "|" if b == zero_bin else " "
        print(f"   {mark}{'█'*h}")
    print()

    p = beats / iters
    print(f"  luck matched or beat the sim in {p:.1%} of reshuffles  (p = {p:.4f})")
    if p < 0.05:
        print("  => PASS. under 5%, so the edge is unlikely to be luck.")
    else:
        print("  => HOLD. too often to rule out luck. need more games.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--rate-season", type=int, default=dt.date.today().year - 1)
    ap.add_argument("--iters", type=int, default=10000)
    main(*vars(ap.parse_args()).values())
