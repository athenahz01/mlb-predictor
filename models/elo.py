"""
models/elo.py
-------------
Pitcher-adjusted MLB Elo, using FiveThirtyEight's published parameters:

  base rating        1500
  home-field adv     +24 Elo
  K-factor           4   (6 in the postseason)
  margin-of-victory  multiplier scales the update by run differential
  rest adjustment    +2.3 Elo per day of rest
  travel adjustment  -(miles ** (1/3)) * 0.31
  pitcher adjustment per-start delta from a rolling "game score" proxy

This is a team-strength baseline with a starter tweak. Like Pythag, its purpose
is to be a floor the simulation must clear at p<0.05 before any version ships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

BASE = 1500.0
HFA = 24.0
K = 4.0
K_POST = 6.0


def expected_home(home_eff: float, away_eff: float) -> float:
    """Elo win expectancy for the home team given effective ratings."""
    return 1.0 / (10 ** ((away_eff - home_eff) / 400.0) + 1.0)


def mov_multiplier(run_diff: int, elo_diff_winner: float) -> float:
    """538-style margin-of-victory multiplier (dampens autocorrelation)."""
    rd = abs(run_diff)
    return math.log(rd + 1.0) * (2.2 / ((elo_diff_winner * 0.001) + 2.2))


def pitcher_adj_from_stats(k: float, bb: float, hr: float, ip: float) -> float:
    """
    Rough per-start pitcher Elo delta from a game-score-like proxy. Positive =
    better than average. Scaled to roughly +-30 Elo for elite/poor starts.
    Replace with a rolling FIP-based rating when you wire real pitcher logs.
    """
    if ip <= 0:
        return 0.0
    k9, bb9, hr9 = 9 * k / ip, 9 * bb / ip, 9 * hr / ip
    # league-ish anchors: K/9 ~8.5, BB/9 ~3.1, HR/9 ~1.2
    score = (k9 - 8.5) * 2.0 - (bb9 - 3.1) * 1.5 - (hr9 - 1.2) * 4.0
    return max(min(score, 35.0), -35.0)


@dataclass
class EloModel:
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, BASE)

    def predict(
        self,
        home: str,
        away: str,
        *,
        home_pitcher_adj: float = 0.0,
        away_pitcher_adj: float = 0.0,
        home_rest: int = 1,
        away_rest: int = 1,
        travel_miles: float = 0.0,
    ) -> float:
        home_eff = self.rating(home) + HFA + home_pitcher_adj + home_rest * 2.3
        away_eff = (
            self.rating(away)
            + away_pitcher_adj
            + away_rest * 2.3
            - (travel_miles ** (1 / 3)) * 0.31
            if travel_miles > 0
            else self.rating(away) + away_pitcher_adj + away_rest * 2.3
        )
        return expected_home(home_eff, away_eff)

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        *,
        postseason: bool = False,
        **predict_kwargs,
    ):
        p_home = self.predict(home, away, **predict_kwargs)
        actual = 1.0 if home_score > away_score else 0.0
        kf = K_POST if postseason else K
        elo_diff = self.rating(home) - self.rating(away)
        winner_diff = elo_diff if actual == 1.0 else -elo_diff
        mult = mov_multiplier(home_score - away_score, winner_diff)
        shift = kf * mult * (actual - p_home)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift
        return p_home

    def run_games(self, games: list[dict]):
        """
        games: chronological list of dicts with at least
          home, away, home_score, away_score
        Optional per-game: home_pitcher_adj, away_pitcher_adj, home_rest,
        away_rest, travel_miles, postseason.
        Returns list of (game, p_home) for backtest scoring.
        """
        preds = []
        for g in games:
            kw = {
                k: g[k]
                for k in (
                    "home_pitcher_adj",
                    "away_pitcher_adj",
                    "home_rest",
                    "away_rest",
                    "travel_miles",
                )
                if k in g
            }
            p = self.update(
                g["home"],
                g["away"],
                g["home_score"],
                g["away_score"],
                postseason=g.get("postseason", False),
                **kw,
            )
            preds.append((g, p))
        return preds


if __name__ == "__main__":
    # quick self-test: feed a season where team A beats team B repeatedly
    m = EloModel()
    games = [{"home": "A", "away": "B", "home_score": 5, "away_score": 3}] * 30 + [
        {"home": "B", "away": "A", "home_score": 2, "away_score": 6}
    ] * 30
    m.run_games(games)
    print("rating A:", round(m.rating("A"), 1))
    print("rating B:", round(m.rating("B"), 1))
    print("P(A home vs B):", round(m.predict("A", "B"), 3))
