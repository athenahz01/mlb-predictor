"""
ingest/pull_statcast.py
-----------------------
Pull Statcast (pitch-level) + FanGraphs season stats via pybaseball, then derive
the per-PA outcome rates the simulator needs.

WINDOWS NOTE (you are on Win11/PowerShell): pybaseball's parallel Statcast pulls
can raise BrokenProcessPool unless the call is under `if __name__ == "__main__"`.
This module keeps all pulls inside functions called from the main guard.

Statcast data is MUTABLE even for past seasons (Tango: each year's ~700k pitches
get revised), so re-pull and version your snapshots; never assume a frozen past.

Usage (run as a script, not imported, for the multiprocessing guard):
  python -m ingest.pull_statcast --season 2025
"""
from __future__ import annotations

import argparse
import datetime as dt

import config

# pybaseball is heavy; import lazily so the rest of the pipeline doesn't need it.
def _pb():
    import pybaseball as pb
    pb.cache.enable()
    return pb


# Map Statcast `events` to our 8 canonical outcomes.
EVENT_MAP = {
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "strikeout": "K", "strikeout_double_play": "K",
}
# everything else with a batted-ball result -> IP_OUT (field_out, grounded_into_dp,
# force_out, sac_fly, sac_bunt, field_error(treated as out for rates), etc.)


def pa_rates_from_statcast(df, who: str = "batter"):
    """
    Given a Statcast dataframe (one row per pitch), collapse to per-PA outcome
    rates for each player. `who` = 'batter' or 'pitcher'.
    """
    import pandas as pd
    # last pitch of each PA carries the `events` result
    pa = df[df["events"].notna()].copy()
    key = "batter" if who == "batter" else "pitcher"
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")
    counts = pa.groupby([key, "outcome"]).size().unstack(fill_value=0)
    for e in config.EVENTS:
        if e not in counts.columns:
            counts[e] = 0
    counts = counts[config.EVENTS]
    rates = counts.div(counts.sum(axis=1), axis=0)
    rates["PA"] = counts.sum(axis=1)
    return rates


def pull_season(season: int):
    pb = _pb()
    start = f"{season}-03-01"
    end = f"{season}-11-30"
    print(f"[statcast] pulling {start}..{end} (this is large; cached after first run)")
    df = pb.statcast(start_dt=start, end_dt=end)
    out = config.SNAPSHOTS / f"statcast_{season}.parquet"
    df.to_parquet(out)
    print(f"[statcast] {len(df):,} pitches -> {out.name}")

    bat = pa_rates_from_statcast(df, "batter")
    pit = pa_rates_from_statcast(df, "pitcher")
    bat.to_parquet(config.SNAPSHOTS / f"pa_rates_batter_{season}.parquet")
    pit.to_parquet(config.SNAPSHOTS / f"pa_rates_pitcher_{season}.parquet")
    print(f"[statcast] derived PA rates: {len(bat)} batters, {len(pit)} pitchers")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    pull_season(args.season)
