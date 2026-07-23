"""
ingest/build_platoon.py
-----------------------
Measure REAL platoon multipliers from league data instead of hand-tuned priors.

For every PA in the statcast snapshot, classify the matchup as platoon-advantage
(batter faces opposite hand, switch hitters always advantaged) or same-hand,
compute per-event rates in each bucket, and express them as multipliers vs the
overall league rate. Writes data/snapshots/platoon_mults.json, which
features/pa_probabilities._platoon_mult picks up automatically.

  python -m ingest.build_platoon --season 2025
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config
from ingest.pull_statcast import EVENT_MAP


def build(season: int):
    df = pd.read_parquet(config.SNAPSHOTS / f"statcast_{season}.parquet",
                         columns=["events", "stand", "p_throws"])
    pa = df[df["events"].notna()].dropna(subset=["stand", "p_throws"]).copy()
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")
    pa["advantage"] = (pa["stand"] == "S") | (pa["stand"] != pa["p_throws"])

    overall = pa["outcome"].value_counts(normalize=True)
    out = {}
    for adv, name in ((True, "advantage"), (False, "same_hand")):
        rates = pa[pa["advantage"] == adv]["outcome"].value_counts(normalize=True)
        out[name] = {e: round(float(rates.get(e, 0) / overall.get(e, 1)), 4)
                     for e in config.EVENTS}
        out[name]["HBP"] = 1.0            # keep HBP neutral; tiny sample noise
    out["_meta"] = {"season": season, "n_pa": int(len(pa)),
                    "share_advantage": round(float(pa["advantage"].mean()), 3)}
    path = config.SNAPSHOTS / "platoon_mults.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[platoon] {len(pa)} PA -> {path.name}")
    for name in ("advantage", "same_hand"):
        print(f"  {name}: " + "  ".join(f"{e} {out[name][e]}" for e in config.EVENTS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    build(ap.parse_args().season)
