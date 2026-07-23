"""Fit base/out play transitions from a frozen Statcast season."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import pandas as pd

import config
from ingest.pull_statcast import EVENT_MAP

COLUMNS = [
    "game_pk",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "events",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "bat_score",
    "post_bat_score",
]


def _bases(row) -> tuple[int, int, int]:
    return (
        int(pd.notna(row["on_1b"])),
        int(pd.notna(row["on_2b"])),
        int(pd.notna(row["on_3b"])),
    )


def _pack(counter: Counter) -> dict:
    total = sum(counter.values())
    return {
        "n": total,
        "transitions": [
            {
                "bases": list(bases),
                "outs_added": outs_added,
                "runs": runs,
                "count": count,
                "probability": count / total,
            }
            for (bases, outs_added, runs), count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def build(season: int) -> dict:
    source = config.SNAPSHOTS / f"statcast_{season}.parquet"
    pitches = pd.read_parquet(source, columns=COLUMNS)
    pa = pitches[pitches["events"].notna()].sort_values(
        ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"]
    )
    states: dict[str, Counter] = defaultdict(Counter)
    fallbacks: dict[str, Counter] = defaultdict(Counter)
    included = 0

    for (_, _inning, _half), frame in pa.groupby(
        ["game_pk", "inning", "inning_topbot"], sort=False
    ):
        rows = list(frame.itertuples(index=False))
        for index, row in enumerate(rows):
            event = EVENT_MAP.get(row.events, "IP_OUT")
            outs_before = int(row.outs_when_up)
            bases_before = _bases(row._asdict())
            post_score = 0 if pd.isna(row.post_bat_score) else float(row.post_bat_score)
            pre_score = 0 if pd.isna(row.bat_score) else float(row.bat_score)
            runs = max(0, int(post_score - pre_score))
            if index + 1 < len(rows):
                following = rows[index + 1]
                outs_after = int(following.outs_when_up)
                bases_after = _bases(following._asdict())
            else:
                outs_after = 3
                bases_after = (0, 0, 0)
            outs_added = max(0, min(3 - outs_before, outs_after - outs_before))
            outcome = (bases_after, outs_added, runs)
            state_key = f"{outs_before}|{''.join(str(value) for value in bases_before)}|{event}"
            states[state_key][outcome] += 1
            fallbacks[event][outcome] += 1
            included += 1

    payload = {
        "_meta": {
            "season": season,
            "source": source.name,
            "n_plays": included,
            "model_version": "empirical-base-out-transitions-v1",
            "minimum_state_plays": 25,
        },
        "states": {key: _pack(counter) for key, counter in states.items()},
        "event_fallbacks": {key: _pack(counter) for key, counter in fallbacks.items()},
    }
    destination = config.SNAPSHOTS / f"transitions_{season}.json"
    destination.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[transitions] {included} plays -> {destination.name}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    build(parser.parse_args().season)
