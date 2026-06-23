"""
backtest/walk_forward.py
------------------------
Walk-forward backtest harness + the significance gate. Skeleton for Stage 4/5:
fill `predict_fn` with your real per-game prediction and `load_games` with a
resolved historical schedule.

Design (matches your house methodology):
  - train/calibrate on seasons <= N, test on N+1 (never random k-fold within a
    season -> that leakage is what produced fake 90%+ MLB models)
  - within the test season, expanding-window day by day
  - score with log-loss / Brier; gate version bumps on paired bootstrap p<0.05
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def paired_bootstrap_pvalue(loss_a: np.ndarray, loss_b: np.ndarray,
                            n_boot: int = 10000, seed: int = 0) -> float:
    """
    One-sided p-value that model A has LOWER mean loss than model B
    (i.e. A is better). Paired by game. This is your ship/hold gate.
    """
    rng = np.random.default_rng(seed)
    diff = loss_b - loss_a                  # >0 where A beats B
    n = len(diff)
    obs = diff.mean()
    if obs <= 0:
        return 1.0                          # A not better -> don't ship
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()
    # p = fraction of bootstrap means <= 0 (no improvement)
    return float(np.mean(boot <= 0))


def logloss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward(load_games: Callable, predict_fn: Callable,
                 test_seasons: list[int]) -> dict:
    """
    load_games(season) -> list of resolved game dicts (must include 'y' truth).
    predict_fn(game)   -> model probability for the binary market under test.
    Returns aggregate scores across the test seasons.
    """
    ps, ys = [], []
    for season in test_seasons:
        for g in load_games(season):
            ps.append(predict_fn(g))
            ys.append(g["y"])
    ps, ys = np.array(ps), np.array(ys, float)
    return {
        "n": len(ys),
        "logloss": float(logloss(ps, ys).mean()),
        "brier": float(np.mean((ps - ys) ** 2)),
        "acc": float(np.mean((ps > 0.5) == (ys == 1))),
    }


def should_ship(candidate_loss: np.ndarray, incumbent_loss: np.ndarray,
                alpha: float = 0.05) -> dict:
    """Ship the candidate model only if it clears the paired-bootstrap gate."""
    p = paired_bootstrap_pvalue(candidate_loss, incumbent_loss)
    return {"p_value": p, "ship": p < alpha,
            "delta_loss": float(incumbent_loss.mean() - candidate_loss.mean())}


if __name__ == "__main__":
    # tiny self-test of the gate
    rng = np.random.default_rng(0)
    incumbent = rng.random(500) * 0.7
    candidate = incumbent - 0.02            # uniformly a bit better
    print(should_ship(candidate, incumbent))
