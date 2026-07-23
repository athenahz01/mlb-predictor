"""Shrunk pitcher pitch-mix versus batter pitch-family skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PitchTypeMatchupModel:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.model_version = str(payload.get("_meta", {}).get("model_version", "unknown"))

    @classmethod
    def from_json(cls, path: Path) -> PitchTypeMatchupModel:
        return cls(json.loads(path.read_text()))

    def factor(
        self,
        *,
        batter_id: int,
        pitcher_id: int,
        batter_side: str,
        pitcher_hand: str,
    ) -> float:
        mix = self.payload.get("pitcher_mix", {}).get(str(pitcher_id), {}).get(f"vs_{batter_side}")
        skill = (
            self.payload.get("batter_skill", {}).get(str(batter_id), {}).get(f"vs_{pitcher_hand}")
        )
        league = self.payload.get("league_value", {}).get(f"vs_{pitcher_hand}")
        if not mix or not skill or not league:
            return 1.0
        expected = sum(
            float(usage) * float(skill.get(pitch_type, league["by_pitch"].get(pitch_type, 0.0)))
            for pitch_type, usage in mix.items()
        )
        baseline = float(league["overall"])
        if baseline <= 0:
            return 1.0
        # Translate wOBA matchup signal conservatively into event-rate scale.
        factor = 1.0 + 0.35 * (expected - baseline) / baseline
        return max(0.90, min(1.10, factor))
