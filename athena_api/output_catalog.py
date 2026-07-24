from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from athena_api.ledger_service import resolve_prediction
from athena_api.models import Prediction


@dataclass(frozen=True)
class OutputDefinition:
    category: str
    statistic: str
    units: str
    value_kind: str
    resolution_field: str
    description: str


_OUTPUTS = (
    # Game
    OutputDefinition("game", "win_probability", "probability", "binary", "winner", "Game winner"),
    OutputDefinition("game", "expected_runs", "runs", "numeric", "runs", "Runs by side or total"),
    OutputDefinition("game", "run_distribution", "runs", "distribution", "runs", "Full run distribution"),
    OutputDefinition("game", "run_line", "probability", "threshold", "run_diff", "Run-line cover"),
    OutputDefinition("game", "team_total", "probability", "threshold", "runs", "Team total"),
    OutputDefinition("game", "f5_winner", "probability", "binary", "f5_winner", "First-five winner"),
    OutputDefinition("game", "f5_total", "probability", "threshold", "f5_total", "First-five total"),
    OutputDefinition("game", "nrfi", "probability", "binary", "first_inning_runs", "No run first inning"),
    OutputDefinition("game", "yrfi", "probability", "binary", "first_inning_runs", "Run first inning"),
    OutputDefinition("game", "first_team_to_score", "probability", "binary", "first_to_score", "First team to score"),
    OutputDefinition("game", "extra_innings", "probability", "binary", "extra_innings", "Game reaches extras"),
    OutputDefinition("game", "shutout", "probability", "binary", "runs", "Team is shut out"),
    OutputDefinition("game", "team_runs_at_least", "probability", "threshold", "runs", "Team scores at least N"),
    # Team
    OutputDefinition("team", "expected_runs", "runs", "numeric", "runs", "Expected team runs"),
    OutputDefinition("team", "run_distribution", "runs", "distribution", "runs", "Team run distribution"),
    OutputDefinition("team", "team_total", "probability", "threshold", "runs", "Team total"),
    OutputDefinition("team", "expected_hits", "hits", "numeric", "hits", "Expected team hits"),
    OutputDefinition("team", "expected_home_runs", "home_runs", "numeric", "home_runs", "Expected team home runs"),
    OutputDefinition("team", "shutout", "probability", "binary", "runs", "Team is shut out"),
    OutputDefinition("team", "runs_at_least", "probability", "threshold", "runs", "Team scores at least N"),
    OutputDefinition("team", "f5_runs", "runs", "numeric", "f5_runs", "First-five offense"),
    OutputDefinition("team", "late_runs", "runs", "numeric", "late_runs", "Sixth inning and later offense"),
    OutputDefinition("team", "bullpen_runs_allowed", "runs", "numeric", "bullpen_runs_allowed", "Bullpen runs allowed"),
    # Pitcher
    OutputDefinition("pitcher", "strikeouts", "strikeouts", "numeric", "strikeouts", "Starter strikeouts"),
    OutputDefinition("pitcher", "strikeout_line", "probability", "threshold", "strikeouts", "Starter strikeout line"),
    OutputDefinition("pitcher", "innings", "innings", "numeric", "innings", "Starter innings"),
    OutputDefinition("pitcher", "batters_faced", "batters_faced", "numeric", "batters_faced", "Batters faced"),
    OutputDefinition("pitcher", "pitches", "pitches", "numeric", "pitches", "Pitch count"),
    OutputDefinition("pitcher", "hits_allowed", "hits", "numeric", "hits_allowed", "Hits allowed"),
    OutputDefinition("pitcher", "walks_allowed", "walks", "numeric", "walks_allowed", "Walks allowed"),
    OutputDefinition("pitcher", "earned_runs", "runs", "numeric", "earned_runs", "Earned runs allowed"),
    OutputDefinition("pitcher", "home_runs_allowed", "home_runs", "numeric", "home_runs_allowed", "Home runs allowed"),
    OutputDefinition("pitcher", "pitcher_win", "probability", "binary", "win", "Pitcher win"),
    OutputDefinition("pitcher", "quality_start", "probability", "binary", "quality_start", "Quality start"),
    # Batter
    OutputDefinition("batter", "hits", "hits", "numeric", "hits", "Batter hits"),
    OutputDefinition("batter", "hit", "probability", "threshold", "hits", "At least one hit"),
    OutputDefinition("batter", "two_plus_hits", "probability", "threshold", "hits", "At least two hits"),
    OutputDefinition("batter", "total_bases", "total_bases", "numeric", "total_bases", "Total bases"),
    OutputDefinition("batter", "total_bases_line", "probability", "threshold", "total_bases", "Total-bases line"),
    OutputDefinition("batter", "home_runs", "home_runs", "numeric", "home_runs", "Expected home runs"),
    OutputDefinition("batter", "home_run", "probability", "threshold", "home_runs", "Home run"),
    OutputDefinition("batter", "run_scored", "probability", "threshold", "runs", "Scores a run"),
    OutputDefinition("batter", "rbi", "probability", "threshold", "rbi", "Records an RBI"),
    OutputDefinition("batter", "walk", "probability", "threshold", "walks", "Records a walk"),
    OutputDefinition("batter", "strikeout", "probability", "threshold", "strikeouts", "Strikes out"),
    OutputDefinition("batter", "stolen_base", "probability", "threshold", "stolen_bases", "Steals a base"),
    OutputDefinition("batter", "plate_appearances", "plate_appearances", "numeric", "plate_appearances", "Plate appearances"),
)

OUTPUT_CATALOG = {(item.category, item.statistic): item for item in _OUTPUTS}


def catalog_payload() -> list[dict[str, str]]:
    return [
        {
            "category": item.category,
            "statistic": item.statistic,
            "units": item.units,
            "value_kind": item.value_kind,
            "resolution_field": item.resolution_field,
            "description": item.description,
        }
        for item in _OUTPUTS
    ]


def _side(payload: dict[str, Any], prediction: Prediction) -> dict[str, Any]:
    side = prediction.parameters.get("side")
    if side in {"home", "away"}:
        return payload[side]
    raise ValueError("prediction requires a home/away side parameter")


def _observed_value(prediction: Prediction, payload: dict[str, Any], field: str) -> Any:
    if prediction.category == "pitcher":
        return payload["pitchers"][str(prediction.player_id)][field]
    if prediction.category == "batter":
        return payload["batters"][str(prediction.player_id)][field]
    if field == "winner":
        return "home" if payload["home"]["runs"] > payload["away"]["runs"] else "away"
    if field == "run_diff":
        difference = payload["home"]["runs"] - payload["away"]["runs"]
        return -difference if prediction.parameters.get("side") == "away" else difference
    if field == "f5_winner":
        home = payload["home"]["f5_runs"]
        away = payload["away"]["f5_runs"]
        return "home" if home > away else "away" if away > home else "tie"
    if field == "f5_total":
        return payload["home"]["f5_runs"] + payload["away"]["f5_runs"]
    if field == "first_inning_runs":
        return payload.get("first_inning_runs", 0)
    if field in {"first_to_score", "extra_innings"}:
        return payload[field]
    if prediction.parameters.get("side") == "total" and field == "runs":
        return payload["home"]["runs"] + payload["away"]["runs"]
    return _side(payload, prediction)[field]


def result_for_prediction(prediction: Prediction, payload: dict[str, Any]) -> dict[str, Any]:
    definition = OUTPUT_CATALOG.get((prediction.category, prediction.statistic))
    if definition is None:
        raise ValueError(f"unsupported output {prediction.category}/{prediction.statistic}")
    actual = _observed_value(prediction, payload, definition.resolution_field)
    result: dict[str, Any] = {"actual": actual}

    if definition.value_kind == "binary":
        stat = prediction.statistic
        side = prediction.parameters.get("side")
        if stat == "win_probability":
            occurred = actual == side
        elif stat == "f5_winner":
            occurred = actual == side
        elif stat == "nrfi":
            occurred = actual == 0
        elif stat == "yrfi":
            occurred = actual > 0
        elif stat == "first_team_to_score":
            occurred = actual == side
        elif stat == "shutout":
            occurred = actual == 0
        else:
            occurred = bool(actual)
        result["outcome"] = int(occurred)
    elif definition.value_kind == "threshold":
        line = float(prediction.parameters["line"])
        direction = prediction.parameters.get("direction", "over")
        if prediction.statistic == "run_line":
            occurred = actual + line > 0
        elif prediction.statistic in {"hit", "two_plus_hits", "home_run", "run_scored", "rbi", "walk", "strikeout", "stolen_base", "runs_at_least", "team_runs_at_least"}:
            occurred = actual >= line
        else:
            occurred = actual > line if direction == "over" else actual < line
        result.update({"line": line, "outcome": int(occurred)})
    return result


def resolve_game_outputs(
    db: Session,
    game_id: str,
    payload: dict[str, Any],
    resolved_at=None,
) -> dict[str, int]:
    rows = list(
        db.scalars(
            select(Prediction).where(
                Prediction.game_id == game_id,
                Prediction.is_headline.is_(True),
                Prediction.resolution_status == "pending",
            )
        )
    )
    resolved = skipped = 0
    for row in rows:
        try:
            result = result_for_prediction(row, payload)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        resolve_prediction(db, row, result, resolved_at)
        resolved += 1
    return {"resolved": resolved, "skipped": skipped}
