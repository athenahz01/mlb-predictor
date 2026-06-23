"""
ingest/pull_fangraphs.py
------------------------
FanGraphs season stats + projection systems (Steamer / ZiPS) via pybaseball.

Use these as BAYESIAN PRIORS for player per-PA rates. Early in the season your
in-sample Statcast rates are tiny and noisy, so blend:

    rate = w * projection + (1 - w) * season_to_date
    w     = PA_prior / (PA_prior + PA_observed)        # shrink toward projection

Marcel (Tom Tango) is the deliberate baseline every projection must beat.

KNOWN ISSUE (open since 2025): pybaseball's batting_stats/pitching_stats scrape
FanGraphs' legacy endpoint, which now returns HTTP 403. This is upstream and
cannot be fixed here. We catch it, warn, and fall back to Baseball-Reference,
which uses a different scrape path. If both fail, the pipeline does NOT crash:
the simulator runs fine on Statcast-derived rates alone (you don't need this
layer mid-season once in-sample PA counts are large).

  python -m ingest.pull_fangraphs --season 2025
"""
from __future__ import annotations

import argparse
import datetime as dt

import config


def _pb():
    import pybaseball as pb
    pb.cache.enable()
    return pb


def _save(df, name: str):
    out = config.SNAPSHOTS / f"{name}.parquet"
    df.to_parquet(out)
    return out


def season_batting(season: int):
    pb = _pb()
    try:
        df = pb.batting_stats(season, qual=1)
        out = _save(df, f"fg_batting_{season}")
        print(f"[fangraphs] batting {season}: {len(df)} players -> {out.name}")
        return df
    except Exception as e:
        print(f"[fangraphs] batting FAILED ({type(e).__name__}): FanGraphs endpoint "
              f"down (known upstream 403). Trying Baseball-Reference fallback...")
        try:
            df = pb.batting_stats_bref(season)
            out = _save(df, f"bref_batting_{season}")
            print(f"[bref] batting {season}: {len(df)} players -> {out.name}")
            return df
        except Exception as e2:
            print(f"[bref] batting fallback also failed ({type(e2).__name__}). "
                  f"Skipping — sim runs on Statcast rates without this.")
            return None


def season_pitching(season: int):
    pb = _pb()
    try:
        df = pb.pitching_stats(season, qual=1)
        out = _save(df, f"fg_pitching_{season}")
        print(f"[fangraphs] pitching {season}: {len(df)} players -> {out.name}")
        return df
    except Exception as e:
        print(f"[fangraphs] pitching FAILED ({type(e).__name__}): FanGraphs endpoint "
              f"down (known upstream 403). Trying Baseball-Reference fallback...")
        try:
            df = pb.pitching_stats_bref(season)
            out = _save(df, f"bref_pitching_{season}")
            print(f"[bref] pitching {season}: {len(df)} players -> {out.name}")
            return df
        except Exception as e2:
            print(f"[bref] pitching fallback also failed ({type(e2).__name__}). "
                  f"Skipping — sim runs on Statcast rates without this.")
            return None


def shrink_rate(projection: float, observed: float,
                pa_observed: int, pa_prior: int = 200) -> float:
    """Regress an observed rate toward a projection prior (see module docstring)."""
    w = pa_prior / (pa_prior + max(pa_observed, 0))
    return w * projection + (1 - w) * observed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    args = ap.parse_args()
    b = season_batting(args.season)
    p = season_pitching(args.season)
    if b is None and p is None:
        print("\n[fangraphs] Both sources down. This is fine — your Statcast pull "
              "already produced per-PA rates for the simulator. Re-try later or "
              "use those directly.")
