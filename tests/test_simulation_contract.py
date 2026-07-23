from __future__ import annotations

import pytest

from features.pa_probabilities import batter_from_slash, pitcher_from_rates
from sim.markov_game import GameContext, Team, run_simulation


def team(code: str) -> Team:
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
    return Team(
        code,
        lineup,
        pitcher_from_rates(f"{code} starter", k_pct=0.23, bb_pct=0.08, hr_pct=0.03),
        pitcher_from_rates(f"{code} pen", k_pct=0.24, bb_pct=0.08, hr_pct=0.03),
    )


def test_simulation_is_seeded_and_probability_contract_holds():
    first = run_simulation(team("H"), team("A"), GameContext(), n_sims=250, seed=19)
    second = run_simulation(team("H"), team("A"), GameContext(), n_sims=250, seed=19)
    assert first["p_home_win"] == second["p_home_win"]
    assert first["exp_total"] == second["exp_total"]
    assert first["p_home_win"] + first["p_away_win"] == pytest.approx(1)
    assert first["p_nrfi"] + first["p_yrfi"] == pytest.approx(1)
    assert sum(first["home_run_distribution"].values()) == pytest.approx(1)
    assert all(0 <= value <= 1 for value in first["total_over"].values())


def test_environment_applies_to_bullpen_matchups():
    neutral = run_simulation(team("H"), team("A"), GameContext(env_hr=1), n_sims=500, seed=8)
    hot = run_simulation(team("H"), team("A"), GameContext(env_hr=1.2), n_sims=500, seed=8)
    assert hot["exp_total"] > neutral["exp_total"]
