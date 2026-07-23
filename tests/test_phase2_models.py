from __future__ import annotations

import numpy as np
import pytest

import config
from features.environment import _air_density, weather_mults
from features.pa_probabilities import Batter, Pitcher, matchup_rates
from models.batter_playing_time import fit_playing_time
from models.pitch_types import PitchTypeMatchupModel
from models.platoon import SplitLine, batting_side, project_platoon
from models.player_projection import SeasonLine, project_player
from models.transitions import TransitionModel
from models.workload import fit_starter_workload
from sim.markov_game import GameContext, Team, run_simulation


def rates(**updates: float) -> dict[str, float]:
    values = dict(config.LEAGUE_PA_RATES)
    values.update(updates)
    total = sum(values.values())
    return {event: value / total for event, value in values.items()}


def test_hierarchical_projection_is_normalized_recency_weighted_and_leakage_safe():
    old = SeasonLine(2024, 500, rates(K=0.30))
    recent = SeasonLine(2025, 500, rates(K=0.16))
    future = SeasonLine(2027, 5_000, rates(K=0.60))
    projected = project_player([old, recent, future], target_season=2026, age=29)
    without_future = project_player([old, recent], target_season=2026, age=29)

    assert sum(projected.rates.values()) == pytest.approx(1.0)
    assert projected.rates == without_future.rates
    assert projected.rates["K"] < config.LEAGUE_PA_RATES["K"]
    assert projected.uncertainty["K"].effective_pa > 0
    assert projected.uncertainty["K"].lower_95 < projected.rates["K"]
    assert projected.uncertainty["K"].upper_95 > projected.rates["K"]


def test_expected_contact_quality_only_changes_contact_events():
    observed = rates(HR=0.02, K=0.28)
    expected = rates(HR=0.06, K=0.10)
    plain = project_player(
        [SeasonLine(2026, 300, observed)],
        target_season=2026,
        age=29,
    )
    contact = project_player(
        [SeasonLine(2026, 300, observed, expected_rates=expected)],
        target_season=2026,
        age=29,
    )
    assert contact.rates["HR"] > plain.rates["HR"]
    # Normalization causes only a small indirect movement; expected K itself is ignored.
    assert abs(contact.rates["K"] - plain.rates["K"]) < 0.01


def test_player_platoon_partial_pooling_and_switch_side():
    league_left = rates(HR=0.025)
    league_right = rates(HR=0.035)
    tiny = project_platoon(
        SplitLine(10, rates(HR=0.15)),
        None,
        league_vs_left=league_left,
        league_vs_right=league_right,
        prior_pa=250,
    )
    large = project_platoon(
        SplitLine(1_000, rates(HR=0.15)),
        None,
        league_vs_left=league_left,
        league_vs_right=league_right,
        prior_pa=250,
    )
    assert abs(tiny.vs_left["HR"] - league_left["HR"]) < abs(
        large.vs_left["HR"] - league_left["HR"]
    )
    assert batting_side("S", "L") == "R"
    assert batting_side("S", "R") == "L"


def test_matchup_uses_player_level_split_without_double_platoon_adjustment():
    batter = Batter(
        "split batter",
        rates(),
        hand="L",
        platoon_rates={"vs_L": rates(HR=0.08), "vs_R": rates(HR=0.02)},
    )
    left = Pitcher("left", rates(), hand="L")
    right = Pitcher("right", rates(), hand="R")
    assert matchup_rates(batter, left)["HR"] > matchup_rates(batter, right)["HR"]


def test_workload_distribution_is_bounded_contextual_and_seeded():
    workload = fit_starter_workload([82, 91, 97, 101, 98], days_rest=5, season_high=101)
    rng_a = np.random.default_rng(12)
    rng_b = np.random.default_rng(12)
    draws_a = [workload.sample_pitch_limit(rng_a) for _ in range(100)]
    draws_b = [workload.sample_pitch_limit(rng_b) for _ in range(100)]
    assert draws_a == draws_b
    assert min(draws_a) >= workload.minimum_pitches
    assert max(draws_a) <= workload.maximum_pitches
    assert workload.probability_starting_inning(3) > workload.probability_starting_inning(7)

    return_workload = fit_starter_workload([95, 100, 102], injury_return=True, season_high=102)
    assert return_workload.expected_pitches <= 75
    assert "injury_return_workload_cap" in return_workload.data_quality_flags


def test_playing_time_survival_is_empirical_and_monotone():
    playing_time = fit_playing_time([3, 4, 4, 5])
    assert playing_time.survival_by_slot_pa[:5] == (1.0, 1.0, 1.0, 0.75, 0.25)
    assert all(
        left >= right
        for left, right in zip(
            playing_time.survival_by_slot_pa,
            playing_time.survival_by_slot_pa[1:],
            strict=False,
        )
    )


def test_playing_time_attribution_does_not_change_team_event_path():
    base_lineup = [Batter(f"B{index}", rates(), order=index + 1) for index in range(9)]
    limited_lineup = [
        Batter(
            f"B{index}",
            rates(),
            order=index + 1,
            playing_time=fit_playing_time([3, 3, 4, 4]),
        )
        for index in range(9)
    ]
    starter = Pitcher("starter", rates(), is_starter=True)
    bullpen = Pitcher("bullpen", rates(), is_starter=False)
    base = Team("B", base_lineup, starter, bullpen)
    limited = Team("L", limited_lineup, starter, bullpen)
    champion = run_simulation(base, base, GameContext(), n_sims=100, seed=91)
    challenger = run_simulation(limited, limited, GameContext(), n_sims=100, seed=91)
    assert challenger["exp_total"] == champion["exp_total"]
    assert challenger["home_batters"][0]["exp_tb"] < champion["home_batters"][0]["exp_tb"]


def test_environment_density_and_roof_fallback():
    assert _air_density(90, 50, 1013.25) < _air_density(50, 50, 1013.25)
    roof = weather_mults("ARI", "2026-07-23")
    assert roof == {"hr": 1.0, "hit": 1.0, "detail": "roof"}


def test_simulation_exposes_workload_distribution_and_bullpen_tiers():
    lineup = [Batter(f"B{index}", rates(), order=index + 1) for index in range(9)]
    starter = Pitcher("starter", rates(K=0.26), is_starter=True)
    aggregate = Pitcher("aggregate", rates(), is_starter=False)
    tiers = (
        Pitcher("low", rates(K=0.18), is_starter=False, role="low"),
        Pitcher("medium", rates(K=0.24), is_starter=False, role="medium"),
        Pitcher("high", rates(K=0.32), is_starter=False, role="high"),
    )
    workload = fit_starter_workload([75, 90, 105, 95, 100])
    team = Team(
        "T",
        lineup,
        starter,
        aggregate,
        starter_workload=workload,
        bullpen_tiers=tiers,
    )
    result = run_simulation(team, team, GameContext(), n_sims=150, seed=44)
    starter_result = result["home_starter_k"]
    assert starter_result["expected_innings"] > 0
    assert set(starter_result["probability_starting_inning"]) == {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    }
    assert 0 < starter_result["expected_pitches"] <= workload.maximum_pitches + 4


def test_empirical_transition_uses_state_then_event_fallback():
    transition = {
        "n": 100,
        "transitions": [
            {
                "bases": [0, 1, 0],
                "outs_added": 0,
                "runs": 1,
                "probability": 1.0,
            }
        ],
    }
    model = TransitionModel(
        {
            "_meta": {"model_version": "test"},
            "states": {"0|100|2B": transition},
            "event_fallbacks": {"2B": transition},
        }
    )
    rng = np.random.default_rng(3)
    exact = model.sample(0, (1, 0, 0), "2B", rng)
    fallback = model.sample(2, (0, 0, 0), "2B", rng)
    assert exact is not None and exact.runs == 1
    assert fallback is not None and fallback.bases == (0, 1, 0)


def test_pitch_type_matchup_is_shrunk_and_bounded():
    model = PitchTypeMatchupModel(
        {
            "_meta": {"model_version": "test"},
            "pitcher_mix": {"7": {"vs_L": {"FF": 0.75, "SL": 0.25}}},
            "batter_skill": {"8": {"vs_R": {"FF": 0.45, "SL": 0.20}}},
            "league_value": {
                "vs_R": {
                    "overall": 0.32,
                    "by_pitch": {"FF": 0.32, "SL": 0.32},
                }
            },
        }
    )
    factor = model.factor(
        batter_id=8,
        pitcher_id=7,
        batter_side="L",
        pitcher_hand="R",
    )
    assert 1.0 < factor <= 1.10
    assert (
        model.factor(
            batter_id=99,
            pitcher_id=7,
            batter_side="L",
            pitcher_hand="R",
        )
        == 1.0
    )
