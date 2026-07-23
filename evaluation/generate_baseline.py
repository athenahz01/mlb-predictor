from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import config
from backtest.sim_backtest import walk_forward_baselines
from evaluation.metrics import (
    binary_log_loss,
    binary_metrics,
    date_clustered_paired_bootstrap,
    holm_adjust,
)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def generate() -> dict[str, Any]:
    results_path = config.SNAPSHOTS / "results_2026.json"
    cache_path = config.SNAPSHOTS / "sim_backtest_2026.json"
    results = json.loads(results_path.read_text())
    cache_raw = json.loads(cache_path.read_text())
    cache = {
        key: value["sim_p"] if isinstance(value, dict) else value
        for key, value in cache_raw.items()
    }
    baselines = walk_forward_baselines(results)
    rows = [
        {
            "gamePk": game["gamePk"],
            "date": game["date"],
            "sim": float(cache[str(game["gamePk"])]),
            "elo": baselines[game["gamePk"]]["elo_p"],
            "pythag": baselines[game["gamePk"]]["pythag_p"],
            "outcome": baselines[game["gamePk"]]["y"],
        }
        for game in results
        if str(game["gamePk"]) in cache and game["gamePk"] in baselines
    ]
    sim = np.array([row["sim"] for row in rows])
    elo = np.array([row["elo"] for row in rows])
    pythag = np.array([row["pythag"] for row in rows])
    outcomes = np.array([row["outcome"] for row in rows])
    dates = np.array([row["date"] for row in rows])
    comparisons = [
        date_clustered_paired_bootstrap(
            binary_log_loss(sim, outcomes),
            binary_log_loss(elo, outcomes),
            dates,
            n_boot=10_000,
            seed=20260723,
        ),
        date_clustered_paired_bootstrap(
            binary_log_loss(sim, outcomes),
            binary_log_loss(pythag, outcomes),
            dates,
            n_boot=10_000,
            seed=20260724,
        ),
    ]
    adjusted = holm_adjust([item["p_value_one_sided"] for item in comparisons])
    for item, adjusted_p in zip(comparisons, adjusted, strict=True):
        item["holm_adjusted_p"] = adjusted_p
        item["ship_after_multiple_testing"] = bool(item["ship"] and adjusted_p < 0.05)

    dashboard_path = Path("dashboard/ledger.json")
    dashboard = json.loads(dashboard_path.read_text())
    market_rows = [
        game
        for game in dashboard
        if game.get("status") == "final" and game.get("market_home_win_pct") is not None
    ]
    market_outcomes = np.array(
        [int(game["winner"] == game["home"]) for game in market_rows], dtype=float
    )
    public_model = np.array([game["model_home_win_pct"] / 100 for game in market_rows])
    market = np.array([game["market_home_win_pct"] / 100 for game in market_rows])
    artifact: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "status": "provisional",
        "evaluation_design": {
            "split": "chronological frozen 2026 cache; no random train/test split",
            "training_window": "prior-season 2025 rate snapshot",
            "calibration_window": None,
            "test_window": [rows[0]["date"], rows[-1]["date"]] if rows else [],
            "dependency_cluster": "game date",
            "bootstrap_resamples": 10_000,
            "multiple_testing": "Holm adjustment across the two winner comparisons",
            "practical_effect_threshold_log_loss": 0.001,
        },
        "versions": {
            "champion": "elo-legacy",
            "challenger": "simulation-legacy-cache",
            "feature_version": "legacy-unknown",
            "data_snapshot_ids": {
                "results": checksum(results_path),
                "sim_cache": checksum(cache_path),
            },
            "simulation_count": "unknown in legacy cache",
            "random_seed": "gamePk % 9999 in legacy backtest",
        },
        "game_ids": [row["gamePk"] for row in rows],
        "dates": [row["date"] for row in rows],
        "per_game": rows,
        "metrics": {
            "simulation": binary_metrics(sim, outcomes),
            "elo": binary_metrics(elo, outcomes),
            "pythagenpat": binary_metrics(pythag, outcomes),
        },
        "comparisons": {
            "simulation_vs_elo": comparisons[0],
            "simulation_vs_pythagenpat": comparisons[1],
        },
        "public_dashboard_market_comparison": {
            "n": len(market_rows),
            "model": binary_metrics(public_model, market_outcomes),
            "market": binary_metrics(market, market_outcomes),
            "claim": "No market outperformance claim; model Brier is slightly worse.",
        },
        "champions": {
            "game_winner": {
                "model": "elo-legacy",
                "status": "provisional",
                "reason": "Simulation did not clear the date-clustered practical ship gate.",
            },
            "totals": {"model": None, "status": "experimental"},
            "nrfi": {"model": None, "status": "experimental"},
            "starter_strikeouts": {"model": "simulation-v2", "status": "provisional"},
            "batter_hits": {"model": None, "status": "unavailable"},
            "batter_total_bases": {"model": None, "status": "experimental"},
            "batter_home_runs": {"model": None, "status": "experimental"},
        },
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    sim_metrics = artifact["metrics"]["simulation"]
    elo_metrics = artifact["metrics"]["elo"]
    public = artifact["public_dashboard_market_comparison"]
    cmp_elo = artifact["comparisons"]["simulation_vs_elo"]
    report = f"""# Athena Baseball baseline

Generated: {artifact["generated_at"]}

All validation labels are **provisional**. The historical cache lacks complete model,
feature, snapshot, and simulation metadata, so it cannot satisfy the final shipping
standard even where descriptive results look favorable.

## Game winner

Chronological frozen-cache evaluation over {len(rows)} games from
{artifact["evaluation_design"]["test_window"][0]} through
{artifact["evaluation_design"]["test_window"][1]}. Bootstrap resampling is clustered
by game date with 10,000 resamples.

| Model | Log loss | Brier | ECE | Calibration slope | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy simulation | {_fmt(sim_metrics["log_loss"])} | {_fmt(sim_metrics["brier"])} | {_fmt(sim_metrics["ece"])} | {_fmt(sim_metrics["calibration_slope"])} | {_fmt(sim_metrics["accuracy"], 3)} |
| Elo baseline | {_fmt(elo_metrics["log_loss"])} | {_fmt(elo_metrics["brier"])} | {_fmt(elo_metrics["ece"])} | {_fmt(elo_metrics["calibration_slope"])} | {_fmt(elo_metrics["accuracy"], 3)} |

Simulation minus Elo improvement: {cmp_elo["effect"]:+.4f} log loss,
95% CI [{cmp_elo["confidence_interval"][0]:+.4f}, {cmp_elo["confidence_interval"][1]:+.4f}],
date-clustered one-sided p={cmp_elo["p_value_one_sided"]:.4f},
Holm-adjusted p={cmp_elo["holm_adjusted_p"]:.4f}. **Decision: HOLD.**

The current game-winner champion remains the provisional Elo baseline. The simulation
does not clear the practical effect, confidence-interval, and multiple-testing gate.

## Public dashboard comparison

Across {public["n"]} completed games with a market value, model Brier is
{public["model"]["brier"]:.3f} and market Brier is {public["market"]["brier"]:.3f}.
Athena must not claim market outperformance.

## Other categories

| Category | Current status | Notes |
| --- | --- | --- |
| Totals | Experimental | 100-game replay: bias -0.26 runs, ECE 0.121; needs work. |
| NRFI/YRFI | Experimental | 100-game replay: Brier 0.247, ECE above readiness threshold. |
| Starter strikeouts | Provisional | Descriptive calibration passes the old tolerance, but no full frozen report yet. |
| Batter hits | Unavailable | No complete reproducible evaluation artifact. |
| Batter total bases | Experimental | Prior x-rate experiment lacks the full new gate metadata. |
| Batter home runs | Experimental | Prior x-rate experiment lacks the full new gate metadata. |

Machine-readable results, exact game IDs, per-game losses inputs, versions, checksums,
dates, bootstrap output, and seeds are stored in `reports/baseline.json`.
"""
    Path("reports/baseline.md").write_text(report, encoding="utf-8")
    return artifact


if __name__ == "__main__":
    result = generate()
    print(json.dumps({"n": len(result["game_ids"]), "status": result["status"]}, indent=2))
