from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from athena_api.database import Base, get_db
from athena_api.main import app
from athena_api.output_catalog import OUTPUT_CATALOG, resolve_game_outputs
from features.pa_probabilities import batter_from_slash, pitcher_from_rates
from pipeline.phase3_outputs import PredictionContext, materialize_simulation, store_simulation
from sim.markov_game import GameContext, Team, run_simulation


def _team(code: str, id_base: int) -> Team:
    lineup = [
        batter_from_slash(
            f"{code}{index}",
            k_pct=0.22,
            bb_pct=0.08,
            hr_pct=0.033,
            order=index + 1,
        )
        for index in range(9)
    ]
    for index, batter in enumerate(lineup):
        batter.mlb_id = id_base + index
    starter = pitcher_from_rates(
        f"{code} starter",
        k_pct=0.24,
        bb_pct=0.08,
        hr_pct=0.03,
    )
    starter.mlb_id = id_base + 100
    bullpen = pitcher_from_rates(
        f"{code} pen",
        k_pct=0.23,
        bb_pct=0.09,
        hr_pct=0.03,
        is_starter=False,
    )
    return Team(code, lineup, starter, bullpen)


@pytest.fixture(scope="module")
def phase3_simulation():
    return run_simulation(
        _team("H", 1_000),
        _team("A", 2_000),
        GameContext(),
        n_sims=180,
        seed=23,
    )


@pytest.fixture
def phase3_context():
    return PredictionContext(
        game_id="A@H-2026-07-24",
        mlb_game_pk=777,
        home_team_id=10,
        away_team_id=20,
        data_snapshot_id="phase3-test",
        simulation_seed=23,
    )


def test_phase3_simulation_exposes_joint_distributions(phase3_simulation):
    simulation = phase3_simulation
    assert simulation["p_f5_home"] + simulation["p_f5_away"] + simulation["p_f5_tie"] == pytest.approx(1)
    assert sum(simulation["f5_total_distribution"].values()) == pytest.approx(1)
    assert sum(simulation["home_team"]["hits_distribution"].values()) == pytest.approx(1)
    assert sum(simulation["home_starter_k"]["outcomes"]["earned_runs"]["distribution"].values()) == pytest.approx(1)
    batter = simulation["home_batters"][0]
    assert batter["exp_pa"] > 0
    assert set(("p_run", "p_rbi", "p_walk", "p_strikeout")) <= batter.keys()
    assert sum(batter["distributions"]["plate_appearances"].values()) == pytest.approx(1)


def test_every_phase3_output_is_materialized_and_labeled(
    phase3_simulation,
    phase3_context,
):
    rows = materialize_simulation(phase3_simulation, phase3_context)
    categories = {row.category for row in rows}
    assert categories == {"game", "team", "pitcher", "batter"}
    assert len(rows) > 300
    assert all((row.category, row.statistic) in OUTPUT_CATALOG for row in rows)
    batter_rows = [row for row in rows if row.category == "batter"]
    assert all(row.player_id is not None for row in batter_rows)
    assert len({row.player_id for row in batter_rows}) == 18
    unavailable = [row for row in rows if row.validation_status == "unavailable"]
    assert {row.statistic for row in unavailable} == {"stolen_base"}
    assert all(
        row.probability is None
        and row.projected_value is None
        and row.resolution_status == "not_applicable"
        for row in unavailable
    )


def test_parameterized_lines_are_distinct_and_full_game_resolves(
    db,
    phase3_simulation,
    phase3_context,
):
    stored = store_simulation(db, phase3_simulation, phase3_context)
    assert stored["created"] > 300
    assert stored["reused"] == 0
    repeated = store_simulation(db, phase3_simulation, phase3_context)
    assert repeated == {"created": 0, "reused": stored["created"]}

    boxscore = {
        "home": {
            "runs": 5,
            "hits": 9,
            "home_runs": 2,
            "f5_runs": 3,
            "late_runs": 2,
            "bullpen_runs_allowed": 1,
        },
        "away": {
            "runs": 3,
            "hits": 7,
            "home_runs": 1,
            "f5_runs": 2,
            "late_runs": 1,
            "bullpen_runs_allowed": 2,
        },
        "first_inning_runs": 0,
        "first_to_score": "home",
        "extra_innings": False,
        "pitchers": {
            "1100": {
                "strikeouts": 7,
                "innings": 6.0,
                "batters_faced": 24,
                "pitches": 96,
                "hits_allowed": 5,
                "walks_allowed": 2,
                "earned_runs": 2,
                "home_runs_allowed": 1,
                "win": True,
                "quality_start": True,
            },
            "2100": {
                "strikeouts": 5,
                "innings": 5.0,
                "batters_faced": 23,
                "pitches": 91,
                "hits_allowed": 7,
                "walks_allowed": 2,
                "earned_runs": 4,
                "home_runs_allowed": 2,
                "win": False,
                "quality_start": False,
            },
        },
        "batters": {
            str(player_id): {
                "hits": 1,
                "total_bases": 2,
                "home_runs": 0,
                "runs": 1,
                "rbi": 1,
                "walks": 0,
                "strikeouts": 1,
                "stolen_bases": 0,
                "plate_appearances": 4,
            }
            for player_id in [*range(1_000, 1_009), *range(2_000, 2_009)]
        },
    }
    resolution = resolve_game_outputs(db, phase3_context.game_id, boxscore)
    assert resolution["skipped"] == 0
    assert resolution["resolved"] == stored["created"] - 18


def test_phase3_catalog_and_family_endpoints(
    phase3_simulation,
    phase3_context,
):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        store_simulation(db, phase3_simulation, phase3_context)
        app.dependency_overrides[get_db] = lambda: db
        try:
            client = TestClient(app)
            catalog = client.get("/api/v1/output-catalog")
            assert catalog.status_code == 200
            assert any(
                row["category"] == "batter" and row["statistic"] == "stolen_base"
                for row in catalog.json()
            )
            for family in ("game", "team", "pitcher", "batter"):
                response = client.get(
                    f"/api/v1/games/{phase3_context.game_id}/predictions/{family}"
                )
                assert response.status_code == 200
                assert response.json()
                assert {row["category"] for row in response.json()} == {family}
                assert all(
                    "supported" in row and "resolution_status" in row
                    for row in response.json()
                )
        finally:
            app.dependency_overrides.clear()
