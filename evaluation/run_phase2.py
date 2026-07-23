"""Chronological, frozen-data evaluation for isolated Phase 2 Tier 1 challengers.

The replay deliberately uses prior-season (2025) Statcast only to construct
features for the opening 2026 test window. Each challenger changes one model
family while all other inputs and simulation seeds remain fixed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from evaluation.challenger import (
    count_crps,
    gate_challenger,
    sha256_file,
    write_experiment_artifact,
)
from evaluation.metrics import binary_log_loss, holm_adjust
from features.load_teams import (
    build_team,
    load_hierarchical_rate_tables,
    load_rate_tables,
    make_bullpen_tiers_from_pitcher_ids,
)
from ingest.pull_statcast import EVENT_MAP
from models.batter_playing_time import fit_playing_time
from models.pitch_types import PitchTypeMatchupModel
from models.transitions import TransitionModel
from models.workload import fit_starter_workload
from sim.markov_game import GameContext, run_simulation

PITCH_COLUMNS = [
    "game_date",
    "game_pk",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "events",
    "stand",
    "p_throws",
    "age_bat",
    "age_pit",
    "post_home_score",
    "post_away_score",
]


def _ordered_unique(values: pd.Series) -> list[int]:
    return [int(value) for value in values.dropna().drop_duplicates().tolist()]


def _game_inputs(frame: pd.DataFrame) -> dict[str, Any] | None:
    frame = frame.sort_values(["at_bat_number", "pitch_number"])
    pa = frame[frame["events"].notna()].copy()
    pa["outcome"] = pa["events"].map(EVENT_MAP).fillna("IP_OUT")
    top = pa[pa["inning_topbot"] == "Top"]
    bottom = pa[pa["inning_topbot"] == "Bot"]
    away_ids = _ordered_unique(top["batter"])[:9]
    home_ids = _ordered_unique(bottom["batter"])[:9]
    if len(home_ids) < 9 or len(away_ids) < 9 or top.empty or bottom.empty:
        return None
    home_starter = int(top.iloc[0]["pitcher"])
    away_starter = int(bottom.iloc[0]["pitcher"])

    batter_actual: dict[int, tuple[int, int]] = {}
    for pid, player_pa in pa.groupby("batter"):
        outcomes = player_pa["outcome"]
        total_bases = (
            (outcomes == "1B").sum()
            + 2 * (outcomes == "2B").sum()
            + 3 * (outcomes == "3B").sum()
            + 4 * (outcomes == "HR").sum()
        )
        batter_actual[int(pid)] = (int((outcomes == "HR").any()), int(total_bases))

    inning_one = frame[frame["inning"] == 1]
    first_inning_runs = int(
        inning_one[["post_home_score", "post_away_score"]].fillna(0).to_numpy().max()
    )
    return {
        "home_lineup": [(pid, str(pid)) for pid in home_ids],
        "away_lineup": [(pid, str(pid)) for pid in away_ids],
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_starter_k": int(((top["pitcher"] == home_starter) & (top["outcome"] == "K")).sum()),
        "away_starter_k": int(
            ((bottom["pitcher"] == away_starter) & (bottom["outcome"] == "K")).sum()
        ),
        "nrfi": int(first_inning_runs == 0),
        "batter_actual": batter_actual,
    }


def _prior_context(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    pa = frame[frame["events"].notna()]
    relievers: dict[str, set[int]] = defaultdict(set)
    pitch_counts: dict[int, list[int]] = defaultdict(list)
    playing_time_counts: dict[int, list[int]] = defaultdict(list)

    for (_, _), half in frame.groupby(["game_pk", "inning_topbot"], sort=False):
        if half.empty:
            continue
        starter = int(half.iloc[0]["pitcher"])
        pitch_counts[starter].append(int((half["pitcher"] == starter).sum()))
        pitching_team = (
            str(half.iloc[0]["home_team"])
            if half.iloc[0]["inning_topbot"] == "Top"
            else str(half.iloc[0]["away_team"])
        )
        relievers[pitching_team].update(
            int(pid) for pid in half["pitcher"].dropna().unique() if int(pid) != starter
        )

    for (_, _), batting_half in pa.groupby(["game_pk", "inning_topbot"], sort=False):
        starters = _ordered_unique(batting_half["batter"])[:9]
        for pid in starters:
            playing_time_counts[pid].append(int((batting_half["batter"] == pid).sum()))

    ages = {}
    for id_col, age_col in (("batter", "age_bat"), ("pitcher", "age_pit")):
        medians = frame.groupby(id_col)[age_col].median().dropna()
        ages.update({int(pid): float(age) for pid, age in medians.items()})
    return {
        "relievers": {team: sorted(ids) for team, ids in relievers.items()},
        "pitch_counts": dict(pitch_counts),
        "playing_time": {
            pid: fit_playing_time(counts) for pid, counts in playing_time_counts.items()
        },
        "ages": ages,
    }


def _base_tables() -> dict:
    loaded = load_rate_tables(2025)
    return {
        "bat": loaded["bat"],
        "pit": loaded["pit"],
        "bhand": loaded["bhand"],
        "phand": loaded["phand"],
        "disable_playing_time": True,
    }


def _teams(game: dict, inputs: dict, tables: dict):
    home = build_team(
        game["home"],
        inputs["home_lineup"],
        inputs["home_starter"],
        str(inputs["home_starter"]),
        tables,
    )
    away = build_team(
        game["away"],
        inputs["away_lineup"],
        inputs["away_starter"],
        str(inputs["away_starter"]),
        tables,
    )
    return home, away


def _simulate_arm(
    name: str,
    game: dict,
    inputs: dict,
    tables: dict,
    prior: dict,
    *,
    n_sims: int,
) -> dict:
    home, away = _teams(game, inputs, tables)
    if name == "bullpen":
        home.bullpen_tiers = make_bullpen_tiers_from_pitcher_ids(
            game["home"], prior["relievers"].get(game["home"], []), tables
        )
        away.bullpen_tiers = make_bullpen_tiers_from_pitcher_ids(
            game["away"], prior["relievers"].get(game["away"], []), tables
        )
    if name == "workload":
        home.starter_workload = fit_starter_workload(
            prior["pitch_counts"].get(inputs["home_starter"], [])
        )
        away.starter_workload = fit_starter_workload(
            prior["pitch_counts"].get(inputs["away_starter"], [])
        )
    if name == "pitch_types":
        pitch_model = prior["pitch_type_model"]
        for offense, defense in ((home, away), (away, home)):
            opposing_pitchers = [defense.starter, *defense.bullpen_tiers]
            for batter in offense.lineup:
                for pitcher in opposing_pitchers:
                    if batter.mlb_id is None or pitcher.mlb_id is None:
                        continue
                    batter_side = (
                        "R"
                        if batter.hand == "S" and pitcher.hand == "L"
                        else "L"
                        if batter.hand == "S"
                        else batter.hand
                    )
                    batter.pitch_type_factors[pitcher.mlb_id] = pitch_model.factor(
                        batter_id=batter.mlb_id,
                        pitcher_id=pitcher.mlb_id,
                        batter_side=batter_side,
                        pitcher_hand=pitcher.hand,
                    )
    context = GameContext(park_code=game["home"])
    if name == "transitions":
        context.transition_model = prior["transition_model"]
    return run_simulation(
        home,
        away,
        context,
        n_sims=n_sims,
        seed=int(game["gamePk"]) % 9_999,
    )


def _losses(
    result: dict,
    game: dict,
    inputs: dict,
    *,
    legacy_prop_share: bool = False,
) -> dict[str, float]:
    y_home = int(game["home_score"] > game["away_score"])
    winner = float(binary_log_loss([result["p_home_win"]], [y_home])[0])
    nrfi = float(binary_log_loss([result["p_nrfi"]], [inputs["nrfi"]])[0])
    total = count_crps(result["total_run_distribution"], game["home_score"] + game["away_score"])
    starter_k = np.mean(
        [
            count_crps(result["home_starter_k"]["distribution"], inputs["home_starter_k"]),
            count_crps(result["away_starter_k"]["distribution"], inputs["away_starter_k"]),
        ]
    )
    hr_errors = []
    tb_errors = []
    for side, lineup_key in (("home", "home_lineup"), ("away", "away_lineup")):
        for index, (pid, _) in enumerate(inputs[lineup_key]):
            actual = inputs["batter_actual"].get(pid)
            if actual is None:
                continue
            prediction = result[f"{side}_batters"][index]
            share = 0.90 if legacy_prop_share else 1.0
            hr_errors.append((prediction["p_hr"] * share - actual[0]) ** 2)
            tb_errors.append((prediction["exp_tb"] * share - actual[1]) ** 2)
    return {
        "winner_log_loss": winner,
        "winner_brier": float((result["p_home_win"] - y_home) ** 2),
        "totals_crps": float(total),
        "totals_absolute_error": float(
            abs(result["exp_total"] - game["home_score"] - game["away_score"])
        ),
        "nrfi_log_loss": nrfi,
        "nrfi_brier": float((result["p_nrfi"] - inputs["nrfi"]) ** 2),
        "starter_k_crps": float(starter_k),
        "batter_hr_brier": float(np.mean(hr_errors)),
        "batter_tb_mse": float(np.mean(tb_errors)),
    }


def run(
    *,
    limit: int = 150,
    n_sims: int = 1_000,
    only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    results_path = config.SNAPSHOTS / "results_2026.json"
    prior_path = config.SNAPSHOTS / "statcast_2025.parquet"
    test_path = config.SNAPSHOTS / "statcast_2026.parquet"
    games = json.loads(results_path.read_text())
    games = sorted(games, key=lambda game: (game["date"], game["gamePk"]))[:limit]
    game_by_id = {int(game["gamePk"]): game for game in games}
    snapshot_ids = {
        "results": sha256_file(results_path),
        "prior_statcast": sha256_file(prior_path),
        "test_statcast": sha256_file(test_path),
    }

    print("[phase2] loading frozen Statcast windows", flush=True)
    prior_frame = pd.read_parquet(prior_path, columns=PITCH_COLUMNS)
    test_frame = pd.read_parquet(
        test_path,
        columns=PITCH_COLUMNS,
        filters=[("game_pk", "in", list(game_by_id))],
    )
    prior = _prior_context(prior_frame)
    prior["transition_model"] = TransitionModel.from_json(
        config.SNAPSHOTS / "transitions_2025.json"
    )
    prior["pitch_type_model"] = PitchTypeMatchupModel.from_json(
        config.SNAPSHOTS / "pitch_types_2025.json"
    )
    test_inputs = {
        int(game_pk): parsed
        for game_pk, frame in test_frame.groupby("game_pk")
        if (parsed := _game_inputs(frame)) is not None
    }

    base = _base_tables()
    loaded_split = load_rate_tables(2025, player_platoon=True)
    split = {**base}
    split["bat_splits"] = loaded_split.get("bat_splits", {})
    split["pit_splits"] = loaded_split.get("pit_splits", {})
    projection = load_hierarchical_rate_tables(2026, (2025,), ages=prior["ages"])
    projection["disable_playing_time"] = True
    projection.pop("bat_splits", None)
    projection.pop("pit_splits", None)
    playing_time = {**base, "disable_playing_time": False, "playing_time": prior["playing_time"]}

    all_experiments = (
        "projection",
        "platoon",
        "workload",
        "bullpen",
        "playing_time",
        "transitions",
        "pitch_types",
    )
    selected = only or all_experiments
    unknown = set(selected) - set(all_experiments)
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}")
    observations: dict[str, list[dict[str, Any]]] = {name: [] for name in selected}
    for game_pk, inputs in test_inputs.items():
        game = game_by_id[game_pk]
        champion_result = _simulate_arm("champion", game, inputs, base, prior, n_sims=n_sims)
        champion_loss = _losses(champion_result, game, inputs, legacy_prop_share=True)
        for name, tables in (
            ("projection", projection),
            ("platoon", split),
            ("workload", base),
            ("bullpen", base),
            ("playing_time", playing_time),
            ("transitions", base),
            ("pitch_types", base),
        ):
            if name not in observations:
                continue
            challenger_result = _simulate_arm(name, game, inputs, tables, prior, n_sims=n_sims)
            challenger_loss = _losses(
                challenger_result,
                game,
                inputs,
                legacy_prop_share=name != "playing_time",
            )
            observations[name].append(
                {
                    "date": game["date"],
                    "game_id": game_pk,
                    "player_ids": {
                        "home_lineup": [pid for pid, _ in inputs["home_lineup"]],
                        "away_lineup": [pid for pid, _ in inputs["away_lineup"]],
                        "home_starter": inputs["home_starter"],
                        "away_starter": inputs["away_starter"],
                    },
                    "champion_loss": champion_loss,
                    "challenger_loss": challenger_loss,
                }
            )
        progress = len(observations[selected[0]])
        if progress % 10 == 0:
            print(
                f"[phase2] completed {progress}/{len(test_inputs)} games",
                flush=True,
            )

    primary = {
        "projection": ("winner_log_loss", 0.001),
        "platoon": ("winner_log_loss", 0.001),
        "workload": ("starter_k_crps", 0.02),
        "bullpen": ("totals_crps", 0.02),
        "playing_time": ("batter_hr_brier", 0.0005),
        "transitions": ("totals_crps", 0.02),
        "pitch_types": ("winner_log_loss", 0.001),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for index, (name, rows) in enumerate(observations.items()):
        metric, threshold = primary[name]
        dates = [row["date"] for row in rows]
        gate = gate_challenger(
            [row["champion_loss"][metric] for row in rows],
            [row["challenger_loss"][metric] for row in rows],
            dates,
            practical_effect=threshold,
            seed=20260723 + index,
        )
        collateral = {}
        for other_metric in (
            "winner_log_loss",
            "winner_brier",
            "totals_crps",
            "totals_absolute_error",
            "nrfi_log_loss",
            "nrfi_brier",
            "starter_k_crps",
            "batter_hr_brier",
            "batter_tb_mse",
        ):
            collateral[other_metric] = {
                "champion": float(np.mean([row["champion_loss"][other_metric] for row in rows])),
                "challenger": float(
                    np.mean([row["challenger_loss"][other_metric] for row in rows])
                ),
            }
        collateral_thresholds = {
            "winner_log_loss": 0.001,
            "winner_brier": 0.001,
            "totals_crps": 0.02,
            "totals_absolute_error": 0.02,
            "nrfi_log_loss": 0.001,
            "nrfi_brier": 0.001,
            "starter_k_crps": 0.02,
            "batter_hr_brier": 0.0005,
            "batter_tb_mse": 0.02,
        }
        materially_damaged = [
            metric_name
            for metric_name, values in collateral.items()
            if values["challenger"] - values["champion"] > collateral_thresholds[metric_name]
        ]
        artifact: dict[str, Any] = {
            "status": "pending_multiple_testing",
            "experiment": name,
            "primary_metric": metric,
            "gate": gate,
            "collateral_metrics": collateral,
            "materially_damaged_outputs": materially_damaged,
            "evaluation_design": {
                "training_window": "2025-03-18 through 2025-09-28 frozen Statcast",
                "calibration_window": None,
                "test_window": [min(dates), max(dates)],
                "dependency_cluster": "game date",
                "bootstrap_resamples": 10_000,
                "simulation_count_per_arm": n_sims,
                "random_seed": "gamePk % 9999",
            },
            "versions": {
                "champion": "simulation-v2-fixed-tier1-inputs",
                "challenger": {
                    "projection": "hierarchical-marcel-v1",
                    "platoon": "player-platoon-partial-pooling-v1",
                    "workload": "starter-workload-v1",
                    "bullpen": "three-leverage-tier-v1",
                    "playing_time": "starter-pa-survival-v1",
                    "transitions": "empirical-base-out-transitions-v1",
                    "pitch_types": "pitch-type-matchup-v1",
                }[name],
                "feature_version": "phase2-tier1-v1",
                "data_snapshot_ids": snapshot_ids,
            },
        }
        summaries[name] = artifact

    adjusted = holm_adjust(
        [summary["gate"]["bootstrap"]["p_value_one_sided"] for summary in summaries.values()]
    )
    for (name, artifact), adjusted_p in zip(summaries.items(), adjusted, strict=True):
        artifact["gate"]["holm_adjusted_p"] = adjusted_p
        artifact["gate"]["passes_holm"] = adjusted_p < 0.05
        artifact["gate"]["ship"] = bool(
            artifact["gate"]["ship"]
            and adjusted_p < 0.05
            and not artifact["materially_damaged_outputs"]
        )
        artifact["status"] = "ship" if artifact["gate"]["ship"] else "reject"
        path = Path("reports/phase2") / f"{name}.json"
        write_experiment_artifact(path, experiment=artifact, observations=observations[name])

    summary_path = Path("reports/phase2/summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--sims", type=int, default=1_000)
    parser.add_argument(
        "--only",
        action="append",
        choices=[
            "projection",
            "platoon",
            "workload",
            "bullpen",
            "playing_time",
            "transitions",
            "pitch_types",
        ],
    )
    arguments = parser.parse_args()
    completed = run(
        limit=arguments.limit,
        n_sims=arguments.sims,
        only=tuple(arguments.only) if arguments.only else None,
    )
    print(
        json.dumps(
            {
                name: {
                    "status": result["status"],
                    "effect": result["gate"]["effect"],
                    "holm_adjusted_p": result["gate"]["holm_adjusted_p"],
                }
                for name, result in completed.items()
            },
            indent=2,
        )
    )
