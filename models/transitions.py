"""Empirical base/out transition distributions fitted from completed plays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlayTransition:
    bases: tuple[int, int, int]
    outs_added: int
    runs: int


class TransitionModel:
    def __init__(self, payload: dict[str, Any], *, minimum_state_plays: int = 25):
        self.payload = payload
        self.minimum_state_plays = minimum_state_plays
        self.model_version = str(payload.get("_meta", {}).get("model_version", "unknown"))

    @classmethod
    def from_json(cls, path: Path) -> TransitionModel:
        return cls(json.loads(path.read_text()))

    @staticmethod
    def _state_key(outs: int, bases: tuple[int, int, int], event: str) -> str:
        return f"{outs}|{''.join(str(value) for value in bases)}|{event}"

    def sample(
        self,
        outs: int,
        bases: tuple[int, int, int],
        event: str,
        rng: np.random.Generator,
    ) -> PlayTransition | None:
        exact = self.payload.get("states", {}).get(self._state_key(outs, bases, event))
        choices = exact if exact and exact.get("n", 0) >= self.minimum_state_plays else None
        if choices is None:
            choices = self.payload.get("event_fallbacks", {}).get(event)
        if not choices:
            return None
        transitions = choices["transitions"]
        probabilities = np.asarray([row["probability"] for row in transitions], dtype=float)
        probabilities /= probabilities.sum()
        row = transitions[int(rng.choice(len(transitions), p=probabilities))]
        raw_bases = row["bases"]
        return PlayTransition(
            bases=(int(raw_bases[0]), int(raw_bases[1]), int(raw_bases[2])),
            outs_added=int(row["outs_added"]),
            runs=int(row["runs"]),
        )
