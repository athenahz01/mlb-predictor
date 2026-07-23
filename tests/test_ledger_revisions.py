from __future__ import annotations

import pytest

from athena_api.ledger_service import create_revision, prediction_tracks, resolve_prediction
from athena_api.schemas import PredictionCreate


def payload(probability: float, reason: str | None = None) -> PredictionCreate:
    return PredictionCreate(
        game_id="NYY@BOS-2026-07-23",
        mlb_game_pk=123,
        category="game",
        statistic="home_win_probability",
        probability=probability,
        model_version="winner-v1",
        git_commit_sha="a" * 40,
        data_snapshot_id="snapshot-1",
        feature_version="features-v1",
        lineup_status="confirmed",
        starting_pitcher_ids={"home": 1, "away": 2},
        confidence="medium",
        revision_reason=reason,
    )


def test_revisions_are_immutable_and_headline_moves(db):
    initial, created = create_revision(db, payload(0.53))
    assert created and initial.revision_number == 1 and initial.is_headline

    repeated, created = create_revision(db, payload(0.53))
    assert not created
    assert repeated.id == initial.id

    latest, created = create_revision(db, payload(0.59, "lineup_confirmation"))
    assert created
    assert latest.revision_number == 2
    assert latest.superseded_prediction_id == initial.id
    assert latest.is_headline
    assert not initial.is_headline
    assert initial.probability == 0.53
    assert initial.validity_status == "superseded"

    tracks = prediction_tracks(db, game_id=initial.game_id)
    assert tracks["initial"][0].id == initial.id
    assert tracks["latest_pregame"][0].id == latest.id


def test_resolution_is_idempotent_but_never_rewritten(db):
    prediction, _ = create_revision(db, payload(0.53))
    resolved = resolve_prediction(db, prediction, {"value": 1})
    assert resolved.final_result == {"value": 1}
    assert resolve_prediction(db, prediction, {"value": 1}).id == prediction.id
    with pytest.raises(ValueError):
        resolve_prediction(db, prediction, {"value": 0})
