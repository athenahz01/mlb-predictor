"""
ingest/xrates.py
----------------
DELUCKED per-PA rates: replace each batted ball's OBSERVED outcome with its
EXPECTED outcome distribution given how it was hit (exit velocity x launch
angle), keeping true-skill events (K, BB, HBP) as observed.

Why: a 105mph/25-degree drive is the same skill whether the wind held it up or
not. Observed rates credit/punish the luck; x-rates credit the contact. This is
the xwOBA idea applied to the full event distribution the simulator needs.

Build the league contact model once per season, then derive x-rates:

  python -m ingest.xrates --season 2025          # writes pa_xrates_{batter,pitcher}_2025.parquet
  python -m ingest.xrates --season 2026

Then predict with them:  run_slate.py --xrates   (loader flag added in load_teams)

The league mapping is a binned EV x LA lookup (5mph x 8deg cells, min 50 balls,
sparse cells fall back to coarser bins), fit on the same season's league data.
Fitting on the season being delucked is fine: the mapping is league-wide (one
player's luck can't move it) and describes physics, not any player's talent.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from ingest.pull_statcast import EVENT_MAP

BIP_EVENTS = ["1B", "2B", "3B", "HR", "IP_OUT"]     # ball-in-play outcomes
EV_BIN = 5.0        # mph
LA_BIN = 8.0        # degrees


def _bin(df):
    ev = (df["launch_speed"] // EV_BIN) * EV_BIN
    la = (df["launch_angle"] // LA_BIN) * LA_BIN
    return ev.astype("Int64").astype(str) + "_" + la.astype("Int64").astype(str)


def fit_contact_model(df: pd.DataFrame) -> dict:
    """League mapping: EV x LA cell -> probability over BIP_EVENTS."""
    pa = df[df["events"].notna()].copy()
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")
    bip = pa[pa["outcome"].isin(BIP_EVENTS)].dropna(
        subset=["launch_speed", "launch_angle"])
    bip["cell"] = _bin(bip)
    tab = (bip.groupby("cell")["outcome"].value_counts().unstack(fill_value=0)
              .reindex(columns=BIP_EVENTS, fill_value=0))
    n = tab.sum(axis=1)
    league = tab.sum(axis=0) / tab.values.sum()
    # regress thin cells toward league BIP distribution (50 pseudo-balls)
    K = 50
    probs = (tab.add(league * K, axis=1)).div(n + K, axis=0)
    return {"cells": probs, "league": league}


def pa_xrates_from_statcast(df: pd.DataFrame, who: str, model: dict):
    """Per-player x-rates: observed K/BB/HBP + expected BIP distribution."""
    key = "batter" if who == "batter" else "pitcher"
    pa = df[df["events"].notna()].copy()
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")

    true_skill = pa[pa["outcome"].isin(["K", "BB", "HBP"])]
    skill_counts = (true_skill.groupby([key, "outcome"]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=["K", "BB", "HBP"], fill_value=0))

    bip = pa[pa["outcome"].isin(BIP_EVENTS)].reset_index(drop=True)
    has_ev = (bip["launch_speed"].notna() & bip["launch_angle"].notna()).values
    # expected distribution for measured balls; league for unmeasured (bunts etc.)
    cells = model["cells"]; league = model["league"]
    exp_vals = np.tile(league.values.astype(float), (len(bip), 1))
    cell_ids = _bin(bip.loc[has_ev])
    matched = cells.reindex(cell_ids.values)
    hit = matched.notna().all(axis=1).values
    rows = np.flatnonzero(has_ev)[hit]
    exp_vals[rows] = matched[hit].values
    exp = pd.DataFrame(exp_vals, columns=BIP_EVENTS)
    exp[key] = bip[key].values
    bip_counts = exp.groupby(key)[BIP_EVENTS].sum()

    counts = skill_counts.join(bip_counts, how="outer").fillna(0.0)
    for e in config.EVENTS:
        if e not in counts.columns:
            counts[e] = 0.0
    counts = counts[config.EVENTS]
    rates = counts.div(counts.sum(axis=1), axis=0)
    rates["PA"] = counts.sum(axis=1).round().astype(int)
    rates.index.name = key
    return rates


def build(season: int):
    path = config.SNAPSHOTS / f"statcast_{season}.parquet"
    df = pd.read_parquet(path, columns=["batter", "pitcher", "events",
                                        "launch_speed", "launch_angle"])
    model = fit_contact_model(df)
    print(f"[xrates] contact model: {len(model['cells'])} EV/LA cells")
    from ingest.pull_statcast import pa_rates_from_statcast
    for who in ("batter", "pitcher"):
        xr = pa_xrates_from_statcast(df, who, model)
        # renormalise so the PA-weighted league mean of every event matches the
        # OBSERVED league mean -- deluck redistributes between players, it must
        # not change the league's run environment.
        obs = pa_rates_from_statcast(df, who)
        common = xr.index.intersection(obs.index)
        w = obs.loc[common, "PA"]
        for e in config.EVENTS:
            xm = float((xr.loc[common, e] * w).sum() / w.sum())
            om = float((obs.loc[common, e] * w).sum() / w.sum())
            if xm > 0:
                xr[e] *= om / xm
        tot = xr[config.EVENTS].sum(axis=1)
        xr[config.EVENTS] = xr[config.EVENTS].div(tot, axis=0)
        out = config.SNAPSHOTS / f"pa_xrates_{who}_{season}.parquet"
        xr.to_parquet(out)
        print(f"[xrates] {who}: {len(xr)} players -> {out.name} (league-mean matched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    build(ap.parse_args().season)
