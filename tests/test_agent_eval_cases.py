from __future__ import annotations

import json
from pathlib import Path

from athena_api.agent import answer_question
from athena_api.ledger_service import create_revision
from athena_api.schemas import PredictionCreate

FIELD_MARKERS = {
    "probability": "%",
    "confidence": "confidence",
    "lineup_status": "Lineup/starter status",
    "last_update": "Updated",
    "model_version": "model",
    "evidence": "Main evidence",
    "uncertainty": "Main uncertainty",
    "data_warning": "Data warning",
}


def seed_supported_outputs(db) -> None:
    outputs = [
        ("game", "home_win_probability", 0.57, None),
        ("game", "total_over_8_5", 0.61, None),
        ("pitcher", "home_starter_strikeouts_over_5_5", 0.64, 101),
        ("batter", "home_run_probability", 0.21, 201),
        ("batter", "hit_probability", 0.68, 202),
        ("batter", "total_bases", None, 203),
    ]
    for category, statistic, probability, player_id in outputs:
        create_revision(
            db,
            PredictionCreate(
                game_id="NYY@BOS-2026-07-23",
                category=category,
                statistic=statistic,
                player_id=player_id,
                probability=probability,
                projected_value=1.8 if statistic == "total_bases" else None,
                model_version=f"{statistic}-v1",
                data_snapshot_id="snapshot-1",
                feature_version="features-v1",
                lineup_status="confirmed",
                confidence="medium" if statistic != "total_bases" else "low",
                data_quality_flags=[],
                evidence={
                    "main_reason": "stored structured evidence",
                    "main_uncertainty": "normal game variance",
                },
            ),
        )


def test_all_agent_cases_are_grounded_or_refuse(db):
    seed_supported_outputs(db)
    cases = json.loads((Path(__file__).parent / "agent_cases.json").read_text())
    for case in cases:
        response = answer_question(
            db,
            case["question"],
            game_id=case.get("game_id"),
        )
        assert response["grounded"], case["id"]
        assert response["tool_calls"][0]["name"] == case["expected_tool"], case["id"]
        answer = response["answer"]
        if case["refuse"]:
            assert "does not have a supported prediction" in answer, case["id"]
        else:
            assert "does not have a supported prediction" not in answer, case["id"]
            for field in case["required_fields"]:
                assert FIELD_MARKERS[field] in answer, (case["id"], field, answer)
        for prohibited in case["prohibited_claims"]:
            assert prohibited.lower() not in answer.lower(), (case["id"], prohibited, answer)
