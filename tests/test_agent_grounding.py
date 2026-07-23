from __future__ import annotations

import json
from pathlib import Path

from athena_api.agent import answer_question
from athena_api.ledger_service import create_revision
from athena_api.schemas import PredictionCreate


def add_winner(db):
    prediction, _ = create_revision(
        db,
        PredictionCreate(
            game_id="NYY@BOS-2026-07-23",
            category="game",
            statistic="home_win_probability",
            probability=0.57,
            model_version="winner-v1",
            data_snapshot_id="snapshot-1",
            feature_version="features-v1",
            lineup_status="confirmed",
            confidence="medium",
            evidence={
                "main_reason": "Boston projects for more baserunners",
                "main_uncertainty": "bullpen availability",
            },
        ),
    )
    return prediction


def test_supported_answer_uses_stored_number_and_required_fields(db):
    add_winner(db)
    response = answer_question(db, "Who is most likely to win?")
    assert "57.0%" in response["answer"]
    assert "winner-v1" in response["answer"]
    assert "confirmed" in response["answer"]
    assert "bullpen availability" in response["answer"]
    assert response["grounded"]


def test_unsupported_request_refuses_without_inventing_a_number(db):
    add_winner(db)
    response = answer_question(db, "Predict the first pitch velocity.")
    assert "does not have a supported prediction" in response["answer"]
    assert "%" not in response["answer"]


def test_agent_evaluation_catalog_has_required_coverage():
    cases = json.loads((Path(__file__).parent / "agent_cases.json").read_text())
    assert len(cases) >= 30
    assert sum(case["refuse"] for case in cases) >= 5
    assert all(case["expected_tool"] for case in cases)
    assert all("prohibited_claims" in case for case in cases)
