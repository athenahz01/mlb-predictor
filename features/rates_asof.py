"""
features/rates_asof.py
----------------------
Point-in-time rate tables: the rates the LIVE model would have had on date D.

    tables = rates_asof("2026-06-15")          # 2025 full season + 2026 games BEFORE Jun 15

This is the backtesting counterpart of load_blended_rate_tables: same blend, same
weights, but the current-season component is truncated at D (exclusive), so a
walk-forward backtest tests exactly the model you deploy, with zero leakage.

Requires data/snapshots/statcast_{cur_season}.parquet (the raw pitch-level pull).
Per-date tables are cached to data/cache/asof/ so a 100-game backtest doesn't
re-derive rates 100 times.
"""
from __future__ import annotations

import pandas as pd

import config
from ingest.pull_statcast import pa_rates_from_statcast
from features.load_teams import load_rate_tables, _blend_tables

CACHE = config.DATA / "cache" / "asof"


def _hand_from_df(df: pd.DataFrame):
    """Dominant handedness from the truncated statcast frame (stand/p_throws)."""
    b = (df.dropna(subset=["stand"]).groupby("batter")["stand"]
           .agg(lambda s: s.mode().iat[0]).to_dict())
    p = (df.dropna(subset=["p_throws"]).groupby("pitcher")["p_throws"]
           .agg(lambda s: s.mode().iat[0]).to_dict())
    return b, p


def rates_asof(date: str, cur_season: int | None = None,
               prior_season: int | None = None, prior_weight: int = 200):
    """Blended rate tables using only information available before `date`."""
    cur_season = cur_season or int(date[:4])
    prior_season = prior_season or cur_season - 1

    CACHE.mkdir(parents=True, exist_ok=True)
    cb = CACHE / f"bat_{date}.parquet"
    cp = CACHE / f"pit_{date}.parquet"

    prior = load_rate_tables(prior_season)

    if cb.exists() and cp.exists():
        bat = pd.read_parquet(cb); pit = pd.read_parquet(cp)
        # hands: current-season mode is stable enough to reuse the full-season cache
        return {"bat": bat, "pit": pit,
                "bhand": prior["bhand"], "phand": prior["phand"]}

    sc_path = config.SNAPSHOTS / f"statcast_{cur_season}.parquet"
    if not sc_path.exists():
        raise FileNotFoundError(
            f"{sc_path} missing - run: python -m ingest.pull_statcast --season {cur_season}")
    df = pd.read_parquet(sc_path,
                         columns=["game_date", "batter", "pitcher", "events",
                                  "stand", "p_throws"])
    df = df[df["game_date"] < date]

    if len(df) == 0:                      # opening day: pure prior
        return prior

    cur_bat = pa_rates_from_statcast(df, "batter")
    cur_pit = pa_rates_from_statcast(df, "pitcher")
    bat = _blend_tables(cur_bat, prior["bat"], prior_weight)
    pit = _blend_tables(cur_pit, prior["pit"], prior_weight)
    bat.to_parquet(cb); pit.to_parquet(cp)

    bhand, phand = _hand_from_df(df)
    return {"bat": bat, "pit": pit,
            "bhand": {**prior["bhand"], **bhand},
            "phand": {**prior["phand"], **phand}}
