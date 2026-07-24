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
from models.platoon import SplitLine, project_platoon


def _event_rates(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["outcome"].value_counts(normalize=True)
    return {event: float(values.get(event, 0.0)) for event in config.EVENTS}


def _split_line(frame: pd.DataFrame) -> SplitLine | None:
    if frame.empty:
        return None
    return SplitLine(pa=len(frame), rates=_event_rates(frame))


def build(season: int, cutoff_date: str | None = None):
    columns = ["game_date", "batter", "pitcher", "events", "stand", "p_throws"]
    df = pd.read_parquet(config.SNAPSHOTS / f"statcast_{season}.parquet", columns=columns)
    if cutoff_date:
        df = df[pd.to_datetime(df["game_date"]).dt.date < pd.Timestamp(cutoff_date).date()]
    pa = df[df["events"].notna()].dropna(subset=["stand", "p_throws"]).copy()
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")
    pa["advantage"] = (pa["stand"] == "S") | (pa["stand"] != pa["p_throws"])

    overall = pa["outcome"].value_counts(normalize=True)
    out = {}
    for adv, name in ((True, "advantage"), (False, "same_hand")):
        rates = pa[pa["advantage"] == adv]["outcome"].value_counts(normalize=True)
        out[name] = {e: round(float(rates.get(e, 0) / overall.get(e, 1)), 4) for e in config.EVENTS}
        out[name]["HBP"] = 1.0  # keep HBP neutral; tiny sample noise
    out["_meta"] = {
        "season": season,
        "n_pa": int(len(pa)),
        "share_advantage": round(float(pa["advantage"].mean()), 3),
    }
    path = config.SNAPSHOTS / "platoon_mults.json"
    path.write_text(json.dumps(out, indent=2))

    league_vs_pitcher = {hand: _event_rates(pa[pa["p_throws"] == hand]) for hand in ("L", "R")}
    league_vs_batter = {hand: _event_rates(pa[pa["stand"] == hand]) for hand in ("L", "R")}
    batters = {}
    for pid, player_pa in pa.groupby("batter", sort=False):
        projection = project_platoon(
            _split_line(player_pa[player_pa["p_throws"] == "L"]),
            _split_line(player_pa[player_pa["p_throws"] == "R"]),
            league_vs_left=league_vs_pitcher["L"],
            league_vs_right=league_vs_pitcher["R"],
        )
        batters[str(int(pid))] = {
            "vs_L": projection.vs_left,
            "vs_R": projection.vs_right,
        }
    pitchers = {}
    for pid, player_pa in pa.groupby("pitcher", sort=False):
        projection = project_platoon(
            _split_line(player_pa[player_pa["stand"] == "L"]),
            _split_line(player_pa[player_pa["stand"] == "R"]),
            league_vs_left=league_vs_batter["L"],
            league_vs_right=league_vs_batter["R"],
        )
        pitchers[str(int(pid))] = {
            "vs_L": projection.vs_left,
            "vs_R": projection.vs_right,
        }
    player_path = config.SNAPSHOTS / f"player_platoon_{season}.json"
    player_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "season": season,
                    "cutoff_date_exclusive": cutoff_date,
                    "n_pa": int(len(pa)),
                    "prior_pa": 250,
                    "model_version": "player-platoon-partial-pooling-v1",
                },
                "batters": batters,
                "pitchers": pitchers,
            },
            separators=(",", ":"),
        )
    )
    print(f"[platoon] {len(pa)} PA -> {path.name}")
    print(f"[platoon] {len(batters)} batters + {len(pitchers)} pitchers -> {player_path.name}")
    for name in ("advantage", "same_hand"):
        print(f"  {name}: " + "  ".join(f"{e} {out[name][e]}" for e in config.EVENTS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--cutoff-date")
    args = ap.parse_args()
    build(args.season, cutoff_date=args.cutoff_date)
