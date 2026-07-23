from __future__ import annotations

from collections import defaultdict
from typing import Any

from athena_api.models import Prediction


def public_evidence(row: Prediction) -> dict[str, Any]:
    evidence = dict(row.evidence or {})
    evidence.pop("_input_fingerprint", None)
    return evidence


def prediction_payload(row: Prediction) -> dict[str, Any]:
    return {
        "id": row.id,
        "game_id": row.game_id,
        "mlb_game_pk": row.mlb_game_pk,
        "category": row.category,
        "statistic": row.statistic,
        "player_id": row.player_id,
        "team_id": row.team_id,
        "probability": row.probability,
        "projected_value": row.projected_value,
        "interval": (
            [row.interval_low, row.interval_high]
            if row.interval_low is not None and row.interval_high is not None
            else None
        ),
        "distribution": row.distribution,
        "model_version": row.model_version,
        "data_snapshot_id": row.data_snapshot_id,
        "lineup_status": row.lineup_status,
        "confidence": row.confidence,
        "validation_status": row.validation_status,
        "data_quality_flags": row.data_quality_flags,
        "evidence": public_evidence(row),
        "revision_number": row.revision_number,
        "revision_reason": row.revision_reason,
        "validity_status": row.validity_status,
        "is_headline": row.is_headline,
        "created_at": row.created_at,
        "resolved_at": row.resolved_at,
    }


def group_games(rows: list[Prediction]) -> list[dict[str, Any]]:
    games: dict[str, list[Prediction]] = defaultdict(list)
    for row in rows:
        games[row.game_id].append(row)
    output = []
    for game_id, predictions in games.items():
        evidence = next((public_evidence(p) for p in predictions if p.evidence), {})
        legacy = evidence.get("legacy_meta", {})
        headline = next(
            (p for p in predictions if p.statistic == "home_win_probability"),
            predictions[0],
        )
        by_stat = {p.statistic: prediction_payload(p) for p in predictions}
        quality_flags = sorted({flag for p in predictions for flag in p.data_quality_flags})
        support_score = max(
            (
                {"high": 0.9, "medium": 0.65, "low": 0.35}.get(p.confidence, 0.35)
                * (1 - min(len(p.data_quality_flags) * 0.15, 0.6))
                for p in predictions
            ),
            default=0,
        )
        output.append(
            {
                "game_id": game_id,
                "mlb_game_pk": headline.mlb_game_pk,
                "away": legacy.get("away", evidence.get("away", "Away")),
                "home": legacy.get("home", evidence.get("home", "Home")),
                "away_name": legacy.get("away_name", legacy.get("away", "Away")),
                "home_name": legacy.get("home_name", legacy.get("home", "Home")),
                "start_time": headline.first_pitch_at,
                "lineup_status": headline.lineup_status,
                "last_updated": max(p.created_at for p in predictions),
                "support_score": round(support_score, 3),
                "data_quality_flags": quality_flags,
                "predictions": by_stat,
            }
        )
    return sorted(output, key=lambda g: (-g["support_score"], g["game_id"]))
