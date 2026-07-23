"""Reusable Evaluation Standard machinery for Phase 2 challengers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import date_clustered_paired_bootstrap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_crps(distribution: Mapping[str, float], actual: int) -> float:
    """Discrete CRPS as the squared distance between forecast and observed CDFs."""
    support = [int(value) for value in distribution]
    upper = max([actual, *support, 0])
    cumulative = 0.0
    score = 0.0
    for value in range(upper + 1):
        cumulative += float(distribution.get(str(value), 0.0))
        observed_cdf = 1.0 if actual <= value else 0.0
        score += (cumulative - observed_cdf) ** 2
    return float(score)


def gate_challenger(
    champion_loss: Sequence[float],
    challenger_loss: Sequence[float],
    dates: Sequence[str],
    *,
    practical_effect: float,
    seed: int,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Gate one preregistered loss metric with time-stability checks."""
    champion = np.asarray(champion_loss, dtype=float)
    challenger = np.asarray(challenger_loss, dtype=float)
    if len(champion) != len(challenger):
        raise ValueError("champion and challenger loss arrays must match")
    bootstrap = date_clustered_paired_bootstrap(
        challenger,
        champion,
        dates,
        n_boot=n_boot,
        seed=seed,
        practical_effect=practical_effect,
    )
    midpoint = len(champion) // 2
    period_effects = [
        float(np.mean(champion[:midpoint] - challenger[:midpoint])),
        float(np.mean(champion[midpoint:] - challenger[midpoint:])),
    ]
    stable = all(effect >= -practical_effect for effect in period_effects)
    return {
        "champion_mean_loss": float(champion.mean()),
        "challenger_mean_loss": float(challenger.mean()),
        "effect": float(np.mean(champion - challenger)),
        "period_effects": period_effects,
        "stable_across_time_halves": stable,
        "bootstrap": bootstrap,
        "ship": bool(bootstrap["ship"] and stable),
    }


def write_experiment_artifact(
    path: Path,
    *,
    experiment: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        **experiment,
        "observations": list(observations),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
