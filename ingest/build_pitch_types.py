"""Build the Tier 2 shrunk pitch-type matchup snapshot."""

from __future__ import annotations

import argparse
import json

import pandas as pd

import config

COLUMNS = ["batter", "pitcher", "pitch_type", "stand", "p_throws", "woba_value"]
MIX_PRIOR_PITCHES = 200.0
SKILL_PRIOR_PA = 100.0


def build(season: int) -> dict:
    source = config.SNAPSHOTS / f"statcast_{season}.parquet"
    frame = pd.read_parquet(source, columns=COLUMNS).dropna(
        subset=["pitch_type", "stand", "p_throws"]
    )
    league_mix_counts = frame.groupby(["stand", "pitch_type"]).size()
    league_mix = league_mix_counts / league_mix_counts.groupby(level=0).transform("sum")
    terminal = frame.dropna(subset=["woba_value"])
    league_pitch_value = terminal.groupby(["p_throws", "pitch_type"])["woba_value"].mean()
    league_overall = terminal.groupby("p_throws")["woba_value"].mean()

    pitcher_mix: dict[str, dict] = {}
    for (pid, side), group in frame.groupby(["pitcher", "stand"], sort=False):
        counts = group["pitch_type"].value_counts()
        prior = league_mix.loc[side]
        pitch_types = counts.index.union(prior.index)
        posterior = {
            pitch_type: (
                float(counts.get(pitch_type, 0))
                + MIX_PRIOR_PITCHES * float(prior.get(pitch_type, 0.0))
            )
            / (len(group) + MIX_PRIOR_PITCHES)
            for pitch_type in pitch_types
        }
        pitcher_mix.setdefault(str(int(pid)), {})[f"vs_{side}"] = posterior

    batter_skill: dict[str, dict] = {}
    for (bid, hand), group in terminal.groupby(["batter", "p_throws"], sort=False):
        league = league_pitch_value.loc[hand]
        values = {}
        for pitch_type in league.index:
            sample = group[group["pitch_type"] == pitch_type]["woba_value"]
            values[pitch_type] = (
                float(sample.sum()) + SKILL_PRIOR_PA * float(league[pitch_type])
            ) / (len(sample) + SKILL_PRIOR_PA)
        batter_skill.setdefault(str(int(bid)), {})[f"vs_{hand}"] = values

    league_value = {
        f"vs_{hand}": {
            "overall": float(league_overall[hand]),
            "by_pitch": {
                pitch_type: float(value)
                for pitch_type, value in league_pitch_value.loc[hand].items()
            },
        }
        for hand in league_overall.index
    }
    payload = {
        "_meta": {
            "season": season,
            "source": source.name,
            "rows": int(len(frame)),
            "terminal_pa": int(len(terminal)),
            "mix_prior_pitches": MIX_PRIOR_PITCHES,
            "skill_prior_pa": SKILL_PRIOR_PA,
            "model_version": "pitch-type-matchup-v1",
        },
        "pitcher_mix": pitcher_mix,
        "batter_skill": batter_skill,
        "league_value": league_value,
    }
    destination = config.SNAPSHOTS / f"pitch_types_{season}.json"
    destination.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"[pitch-types] {len(pitcher_mix)} pitchers + {len(batter_skill)} batters "
        f"-> {destination.name}"
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    build(parser.parse_args().season)
