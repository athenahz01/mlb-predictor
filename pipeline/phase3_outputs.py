from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from athena_api.ledger_service import create_revision
from athena_api.schemas import PredictionCreate


@dataclass(frozen=True)
class PredictionContext:
    game_id: str
    mlb_game_pk: int
    home_team_id: int
    away_team_id: int
    data_snapshot_id: str
    first_pitch_at: dt.datetime | None = None
    lineup_status: str = "confirmed"
    model_version: str = "markov-game-v3"
    feature_version: str = "phase3-v1"
    rate_source_version: str | None = None
    git_commit_sha: str | None = None
    simulation_seed: int = 0
    context_feature_flags: list[str] = field(default_factory=list)


def _interval(distribution: dict[str, float] | None) -> tuple[float | None, float | None]:
    if not distribution:
        return None, None
    ordered = sorted((float(value), probability) for value, probability in distribution.items())
    cumulative = 0.0
    low = ordered[0][0]
    high = ordered[-1][0]
    found_low = False
    for value, probability in ordered:
        cumulative += probability
        if not found_low and cumulative >= 0.1:
            low = value
            found_low = True
        if cumulative >= 0.9:
            high = value
            break
    return low, high


def materialize_simulation(
    simulation: dict[str, Any],
    context: PredictionContext,
) -> list[PredictionCreate]:
    rows: list[PredictionCreate] = []
    lineup_ids = {
        "home": [item["player_id"] for item in simulation["home_batters"]],
        "away": [item["player_id"] for item in simulation["away_batters"]],
    }
    starter_ids = {
        "home": simulation["home_starter_k"]["player_id"],
        "away": simulation["away_starter_k"]["player_id"],
    }
    if context.lineup_status == "confirmed":
        if any(player_id is None for side in lineup_ids.values() for player_id in side):
            raise ValueError("confirmed lineups require an MLB ID for every batter")
        if any(player_id is None for player_id in starter_ids.values()):
            raise ValueError("confirmed starting pitchers require MLB IDs")
    common = {
        "game_id": context.game_id,
        "mlb_game_pk": context.mlb_game_pk,
        "model_version": context.model_version,
        "git_commit_sha": context.git_commit_sha,
        "data_snapshot_id": context.data_snapshot_id,
        "rate_source_version": context.rate_source_version,
        "feature_version": context.feature_version,
        "simulation_settings": {"n_sims": simulation["n_sims"]},
        "simulation_seed": context.simulation_seed,
        "simulation_seed_policy": "explicit deterministic seed",
        "lineup_player_ids": lineup_ids,
        "lineup_status": context.lineup_status,
        "starting_pitcher_ids": starter_ids,
        "context_feature_flags": context.context_feature_flags,
        "first_pitch_at": context.first_pitch_at,
    }

    def add(
        category: str,
        statistic: str,
        *,
        probability: float | None = None,
        projected_value: float | None = None,
        distribution: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        units: str,
        player_id: int | None = None,
        team_id: int | None = None,
        validation_status: str = "experimental",
        confidence: str = "low",
        data_quality_flags: Iterable[str] = (),
        evidence_note: str | None = None,
    ) -> None:
        low, high = _interval(distribution)
        flags = sorted(set([*simulation.get("data_quality_flags", []), *data_quality_flags]))
        rows.append(
            PredictionCreate(
                **common,
                category=category,
                statistic=statistic,
                probability=probability,
                projected_value=projected_value,
                interval_low=low,
                interval_high=high,
                distribution=distribution,
                parameters=parameters or {},
                units=units,
                player_id=player_id,
                team_id=team_id,
                validation_status=validation_status,
                confidence=confidence,
                data_quality_flags=flags,
                evidence={
                    "method": "joint event-driven Monte Carlo",
                    "n_sims": simulation["n_sims"],
                    "note": evidence_note
                    or "Experimental Phase 3 output; not a market-outperformance claim.",
                },
            )
        )

    # Game outputs
    add(
        "game",
        "win_probability",
        probability=simulation["p_home_win"],
        parameters={"side": "home"},
        units="probability",
        team_id=context.home_team_id,
        validation_status="provisional",
        confidence="medium",
    )
    add(
        "game",
        "win_probability",
        probability=simulation["p_away_win"],
        parameters={"side": "away"},
        units="probability",
        team_id=context.away_team_id,
        validation_status="provisional",
        confidence="medium",
    )
    for side, team_id, mean_key, dist_key in (
        ("home", context.home_team_id, "exp_home_runs", "home_run_distribution"),
        ("away", context.away_team_id, "exp_away_runs", "away_run_distribution"),
        ("total", None, "exp_total", "total_run_distribution"),
    ):
        add(
            "game",
            "expected_runs",
            projected_value=simulation[mean_key],
            distribution=simulation[dist_key],
            parameters={"side": side},
            units="runs",
            team_id=team_id,
        )
        add(
            "game",
            "run_distribution",
            projected_value=simulation[mean_key],
            distribution=simulation[dist_key],
            parameters={"side": side},
            units="runs",
            team_id=team_id,
        )
    for key, probability in simulation["run_lines"].items():
        side, line_text = key.split("_", 1)
        add(
            "game",
            "run_line",
            probability=probability,
            parameters={"side": side, "line": float(line_text), "direction": "over"},
            units="probability",
            team_id=context.home_team_id if side == "home" else context.away_team_id,
        )
    for side, team_id, grid_key in (
        ("home", context.home_team_id, "home_team_total_over"),
        ("away", context.away_team_id, "away_team_total_over"),
    ):
        for key, probability in simulation[grid_key].items():
            add(
                "game",
                "team_total",
                probability=probability,
                parameters={"side": side, "line": float(key.removeprefix("over_")), "direction": "over"},
                units="probability",
                team_id=team_id,
            )
    for side, team_id, probability in (
        ("home", context.home_team_id, simulation["p_f5_home"]),
        ("away", context.away_team_id, simulation["p_f5_away"]),
        ("tie", None, simulation["p_f5_tie"]),
    ):
        add(
            "game",
            "f5_winner",
            probability=probability,
            parameters={"side": side},
            units="probability",
            team_id=team_id,
        )
    for key, probability in simulation["f5_total_over"].items():
        add(
            "game",
            "f5_total",
            probability=probability,
            projected_value=simulation["exp_f5_total"],
            distribution=simulation["f5_total_distribution"],
            parameters={"side": "total", "line": float(key.removeprefix("over_")), "direction": "over"},
            units="probability",
        )
    for statistic, probability in (
        ("nrfi", simulation["p_nrfi"]),
        ("yrfi", simulation["p_yrfi"]),
        ("extra_innings", simulation["p_extra_innings"]),
    ):
        add("game", statistic, probability=probability, units="probability")
    for side, team_id, probability in (
        ("home", context.home_team_id, simulation["p_first_to_score_home"]),
        ("away", context.away_team_id, simulation["p_first_to_score_away"]),
    ):
        add(
            "game",
            "first_team_to_score",
            probability=probability,
            parameters={"side": side},
            units="probability",
            team_id=team_id,
        )
    for side, team_id, dist_key in (
        ("home", context.home_team_id, "home_run_distribution"),
        ("away", context.away_team_id, "away_run_distribution"),
    ):
        shutout = simulation[f"p_{side}_shutout"]
        add(
            "game",
            "shutout",
            probability=shutout,
            parameters={"side": side},
            units="probability",
            team_id=team_id,
        )
        distribution = simulation[dist_key]
        for threshold in range(1, 9):
            probability = sum(
                mass for value, mass in distribution.items() if int(value) >= threshold
            )
            add(
                "game",
                "team_runs_at_least",
                probability=probability,
                parameters={"side": side, "line": threshold},
                units="probability",
                team_id=team_id,
            )

    # Team outputs
    for side, team_id, mean_key, dist_key in (
        ("home", context.home_team_id, "exp_home_runs", "home_run_distribution"),
        ("away", context.away_team_id, "exp_away_runs", "away_run_distribution"),
    ):
        team = simulation[f"{side}_team"]
        base = {"side": side}
        add("team", "expected_runs", projected_value=simulation[mean_key], distribution=simulation[dist_key], parameters=base, units="runs", team_id=team_id)
        add("team", "run_distribution", projected_value=simulation[mean_key], distribution=simulation[dist_key], parameters=base, units="runs", team_id=team_id)
        add("team", "expected_hits", projected_value=team["expected_hits"], distribution=team["hits_distribution"], parameters=base, units="hits", team_id=team_id)
        add("team", "expected_home_runs", projected_value=team["expected_home_runs"], distribution=team["home_runs_distribution"], parameters=base, units="home_runs", team_id=team_id)
        add("team", "shutout", probability=simulation[f"p_{side}_shutout"], parameters=base, units="probability", team_id=team_id)
        add("team", "runs_at_least", probability=simulation[f"p_{side}_5plus_runs"], parameters={**base, "line": 5}, units="probability", team_id=team_id)
        add("team", "f5_runs", projected_value=team["expected_f5_runs"], distribution=team["f5_run_distribution"], parameters=base, units="runs", team_id=team_id)
        add("team", "late_runs", projected_value=team["expected_late_runs"], distribution=team["late_run_distribution"], parameters=base, units="runs", team_id=team_id)
        add("team", "bullpen_runs_allowed", projected_value=team["expected_bullpen_runs_allowed"], distribution=team["bullpen_runs_allowed_distribution"], parameters=base, units="runs", team_id=team_id)
        for key, probability in simulation[f"{side}_team_total_over"].items():
            add("team", "team_total", probability=probability, parameters={**base, "line": float(key.removeprefix("over_")), "direction": "over"}, units="probability", team_id=team_id)

    # Starting pitchers
    for side, team_id in (
        ("home", context.home_team_id),
        ("away", context.away_team_id),
    ):
        pitcher = simulation[f"{side}_starter_k"]
        player_id = pitcher["player_id"]
        flags = pitcher["data_quality_flags"]
        add("pitcher", "strikeouts", projected_value=pitcher["mean"], distribution=pitcher["distribution"], parameters={"side": side}, units="strikeouts", player_id=player_id, team_id=team_id, validation_status="provisional", confidence="medium", data_quality_flags=flags)
        for key, probability in pitcher["over"].items():
            add("pitcher", "strikeout_line", probability=probability, parameters={"side": side, "line": float(key.removeprefix("over_")), "direction": "over"}, units="probability", player_id=player_id, team_id=team_id, validation_status="provisional", confidence="medium", data_quality_flags=flags)
        for statistic, key, dist_key, units in (
            ("innings", "expected_innings", "innings", "innings"),
            ("batters_faced", "expected_batters_faced", "batters_faced", "batters_faced"),
            ("pitches", "expected_pitches", "pitches", "pitches"),
        ):
            add("pitcher", statistic, projected_value=pitcher[key], distribution=pitcher["workload_distributions"][dist_key], parameters={"side": side}, units=units, player_id=player_id, team_id=team_id, data_quality_flags=flags)
        for statistic, units in (
            ("hits_allowed", "hits"),
            ("walks_allowed", "walks"),
            ("earned_runs", "runs"),
            ("home_runs_allowed", "home_runs"),
        ):
            outcome = pitcher["outcomes"][statistic]
            add("pitcher", statistic, projected_value=outcome["mean"], distribution=outcome["distribution"], parameters={"side": side}, units=units, player_id=player_id, team_id=team_id, data_quality_flags=flags)
        add("pitcher", "pitcher_win", probability=pitcher["p_pitcher_win"], parameters={"side": side}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=[*flags, "pitcher_win_proxy"])
        add("pitcher", "quality_start", probability=pitcher["p_quality_start"], parameters={"side": side}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=flags)

    # Every confirmed lineup slot is materialized independently with its MLB ID.
    for side, team_id in (
        ("home", context.home_team_id),
        ("away", context.away_team_id),
    ):
        for batter in simulation[f"{side}_batters"]:
            player_id = batter["player_id"]
            flags = batter["data_quality_flags"]
            distributions = batter["distributions"]
            add("batter", "hits", projected_value=batter["exp_hits"], distribution=distributions["hits"], parameters={"side": side}, units="hits", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add("batter", "hit", probability=batter["p_hit"], parameters={"side": side, "line": 1}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add("batter", "two_plus_hits", probability=batter["p_2plus_hits"], parameters={"side": side, "line": 2}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add("batter", "total_bases", projected_value=batter["exp_tb"], distribution=distributions["total_bases"], parameters={"side": side}, units="total_bases", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            for key, probability in batter["total_bases_over"].items():
                add("batter", "total_bases_line", probability=probability, parameters={"side": side, "line": float(key.removeprefix("over_")), "direction": "over"}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add("batter", "home_runs", projected_value=batter["exp_hr"], distribution=distributions["home_runs"], parameters={"side": side}, units="home_runs", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            for statistic, probability, field_name in (
                ("home_run", batter["p_hr"], "home_runs"),
                ("run_scored", batter["p_run"], "runs"),
                ("rbi", batter["p_rbi"], "rbi"),
                ("walk", batter["p_walk"], "walks"),
                ("strikeout", batter["p_strikeout"], "strikeouts"),
            ):
                add("batter", statistic, probability=probability, distribution=distributions[field_name], parameters={"side": side, "line": 1}, units="probability", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add("batter", "plate_appearances", projected_value=batter["exp_pa"], distribution=distributions["plate_appearances"], parameters={"side": side}, units="plate_appearances", player_id=player_id, team_id=team_id, data_quality_flags=flags)
            add(
                "batter",
                "stolen_base",
                parameters={"side": side, "line": 1},
                units="probability",
                player_id=player_id,
                team_id=team_id,
                validation_status="unavailable",
                data_quality_flags=[*flags, "stolen_base_model_unavailable"],
                evidence_note="The simulator does not yet model stolen-base attempts; no probability is fabricated.",
            )
    return rows


def store_simulation(
    db: Session,
    simulation: dict[str, Any],
    context: PredictionContext,
) -> dict[str, int]:
    created = reused = 0
    for payload in materialize_simulation(simulation, context):
        _, was_created = create_revision(db, payload)
        created += int(was_created)
        reused += int(not was_created)
    return {"created": created, "reused": reused}
