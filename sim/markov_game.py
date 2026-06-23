"""
sim/markov_game.py
------------------
Event-driven Monte Carlo over the 24 base-out states (3 out-counts x 8 base
configurations). This is the GENERATIVE CORE: one simulation yields the joint
distribution of every target you care about, all internally consistent:

  game-level : P(home win), run total dist (over/under), run line, F5, NRFI/first-inning
  team       : runs per side, first-to-score
  batter     : hits, total bases, HR, P(>=1 HR), P(>=2 hits)
  pitcher    : strikeouts (and their distribution -> P(over k.5))

Why event-driven MC rather than a closed-form transition matrix: the matrix gives
you analytic run distributions but the event sim lets you attribute every PA to a
batter and every K to a pitcher, which is what the prop side of the brief needs.

Base-running uses standard simplifying assumptions (documented inline). They are
deliberately conservative and are the first thing to refine against Retrosheet
empirical advancement rates.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import config
from features.pa_probabilities import Batter, Pitcher, matchup_rates

EVENTS = config.EVENTS


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class Team:
    code: str
    lineup: List[Batter]               # 9 batters, in order
    starter: Pitcher
    bullpen: Pitcher                   # single aggregate reliever profile (v1)

    def __post_init__(self):
        assert len(self.lineup) == 9, "lineup must be 9 batters"


@dataclass
class GameContext:
    park_code: str = "_DEFAULT"
    ump_k_mult: float = 1.0
    innings: int = 9
    hfa_mult: float = 1.045            # small home-offense bump -> ~53% home win baseline


# --------------------------------------------------------------------------
# Pre-compute per-(batter,pitcher) PA probability vectors so we sample fast.
# We build two matchup tables per team: vs the opposing starter and vs the
# opposing bullpen. Times-through-order is applied as a small additive bump.
# --------------------------------------------------------------------------
def _precompute(team: Team, opp_starter: Pitcher, opp_pen: Pitcher,
                ctx: GameContext):
    """Return dict: batter_idx -> {'SP': probs, 'SP_tto': probs, 'BP': probs}."""
    table = {}
    for i, bat in enumerate(team.lineup):
        vs_sp = matchup_rates(bat, opp_starter, ctx.park_code, ctx.ump_k_mult,
                              tto_bump=0.0)
        vs_sp_tto = matchup_rates(bat, opp_starter, ctx.park_code, ctx.ump_k_mult,
                                  tto_bump=config.TTO_PENALTY)
        vs_bp = matchup_rates(bat, opp_pen, ctx.park_code, ctx.ump_k_mult,
                              tto_bump=0.0)
        table[i] = {
            "SP": np.array([vs_sp[e] for e in EVENTS]),
            "SP_tto": np.array([vs_sp_tto[e] for e in EVENTS]),
            "BP": np.array([vs_bp[e] for e in EVENTS]),
        }
    return table


# event indices
iBB, iHBP, i1B, i2B, i3B, iHR, iK, iOUT = range(8)


@dataclass
class _Tally:
    runs: int = 0
    hits: np.ndarray = None            # per-batter hit count
    tb: np.ndarray = None              # per-batter total bases
    hr: np.ndarray = None              # per-batter HR
    k_sp: int = 0                      # strikeouts by the starter
    k_bp: int = 0


def _advance(bases, event, rng):
    """
    Given occupied bases (b1,b2,b3 as 0/1) and a non-out event, return
    (new_bases, runs_scored). Batter always ends on the correct base; runner
    extra-advancement is probabilistic and handled here (not by the caller).

    Advancement probabilities are league-ballpark and are the first thing to
    refine against Retrosheet empirical base-running rates.
    """
    b1, b2, b3 = bases
    if event == iHR:
        return (0, 0, 0), 1 + b1 + b2 + b3
    if event == i3B:
        return (0, 0, 1), b1 + b2 + b3          # batter to 3rd, all score
    if event == i2B:
        # batter -> 2nd; runners on 2nd/3rd score; 1st scores ~40%, else -> 3rd
        runs = b2 + b3
        new_b3 = 0
        if b1:
            if rng.random() < 0.40:
                runs += 1
            else:
                new_b3 = 1
        return (0, 1, new_b3), runs
    if event == i1B:
        # batter -> 1st; 3rd scores; 2nd scores ~60% else -> 3rd;
        # 1st -> 2nd (or 3rd ~30%)
        runs = b3
        new_b2 = 0
        new_b3 = 0
        if b2:
            if rng.random() < 0.60:
                runs += 1
            else:
                new_b3 = 1
        if b1:
            if rng.random() < 0.30 and new_b3 == 0:
                new_b3 = 1
            else:
                new_b2 = 1
        return (1, new_b2, new_b3), runs
    if event in (iBB, iHBP):
        # force advance only
        if b1 and b2 and b3:
            return (1, 1, 1), 1
        if b1 and b2:
            return (1, 1, 1), 0
        if b1:
            return (1, 1, b3), 0
        return (1, b2, b3), 0
    return bases, 0


def _simulate_half(lineup_probs, lineup_start_idx, pitch_state, tally,
                   pitcher_is_sp, rng, ghost_on_second=False):
    """
    Simulate one half-inning. Mutates `tally`. Returns (runs, next_lineup_idx).
    pitch_state: dict tracking starter pitches/innings to decide bullpen.
    """
    outs = 0
    b1, b2, b3 = (0, 0, 1) if ghost_on_second else (0, 0, 0)
    runs = 0
    idx = lineup_start_idx
    times_through = pitch_state["batters_faced"] // 9

    while outs < 3:
        use_bp = pitch_state["bullpen_in"]
        if not use_bp:
            key = "SP_tto" if times_through >= 2 else "SP"
        else:
            key = "BP"
        probs = lineup_probs[idx][key]
        ev = rng.choice(8, p=probs)

        # pitch-count proxy: ~3.8 pitches per PA
        pitch_state["pitches"] += 3.8
        pitch_state["batters_faced"] += 1
        times_through = pitch_state["batters_faced"] // 9

        # record pitcher strikeouts
        if ev == iK:
            if pitch_state["bullpen_in"]:
                tally.k_bp += 1
            else:
                tally.k_sp += 1

        # batter box-score
        if ev in (i1B, i2B, i3B, iHR):
            tally.hits[idx] += 1
            tally.tb[idx] += {i1B: 1, i2B: 2, i3B: 3, iHR: 4}[ev]
            if ev == iHR:
                tally.hr[idx] += 1

        if ev == iK or ev == iOUT:
            # IP_OUT: small chance of a productive out (sac fly / advance)
            if ev == iOUT and outs < 2 and b3 and rng.random() < 0.25:
                runs += 1
                b3 = 0
            # simple double-play: runner on 1st, <2 outs, grounder-ish
            if ev == iOUT and b1 and outs < 2 and rng.random() < 0.12:
                outs += 1
                b1 = 0
            outs += 1
        else:
            (b1, b2, b3), scored = _advance((b1, b2, b3), ev, rng)
            runs += scored

        idx = (idx + 1) % 9

        # bullpen hook (checked after each PA)
        if (not pitch_state["bullpen_in"] and pitcher_is_sp and
                (pitch_state["pitches"] >= config.STARTER_PITCH_LIMIT or
                 pitch_state["inning"] >= config.STARTER_IP_SOFT_CAP)):
            pitch_state["bullpen_in"] = True

    return runs, idx


def simulate_game(home: Team, away: Team, ctx: GameContext, rng) -> dict:
    """Simulate ONE game. Returns a dict of outcomes for this single realisation."""
    home_probs = _precompute(home, away.starter, away.bullpen, ctx)
    away_probs = _precompute(away, home.starter, home.bullpen, ctx)

    # home-field advantage: nudge home offensive events up slightly, renormalise
    if ctx.hfa_mult != 1.0:
        for i in home_probs:
            for key in home_probs[i]:
                v = home_probs[i][key].copy()
                for e_idx in (iBB, i1B, i2B, i3B, iHR):
                    v[e_idx] *= ctx.hfa_mult
                home_probs[i][key] = v / v.sum()

    h_t = _Tally(hits=np.zeros(9, int), tb=np.zeros(9, int), hr=np.zeros(9, int))
    a_t = _Tally(hits=np.zeros(9, int), tb=np.zeros(9, int), hr=np.zeros(9, int))

    h_pitch = {"pitches": 0, "batters_faced": 0, "bullpen_in": False, "inning": 0}
    a_pitch = {"pitches": 0, "batters_faced": 0, "bullpen_in": False, "inning": 0}

    h_idx = a_idx = 0
    home_runs = away_runs = 0
    f5_home = f5_away = 0          # first-five-innings runs
    inning1_runs = 0
    first_to_score = None         # 'H' / 'A'

    inning = 1
    while True:
        extras = inning > ctx.innings
        ghost = config.GHOST_RUNNER_EXTRAS and extras

        # ---- top: away bats, faces home pitching ----
        h_pitch["inning"] = inning
        r, a_idx = _simulate_half(away_probs, a_idx, h_pitch, a_t,
                                  pitcher_is_sp=not h_pitch["bullpen_in"], rng=rng,
                                  ghost_on_second=ghost)
        away_runs += r
        if inning <= 5: f5_away += r
        if inning == 1: inning1_runs += r
        if first_to_score is None and r > 0: first_to_score = "A"

        # walk-off short-circuit: home already leads, skip bottom of 9th+
        if inning >= ctx.innings and home_runs > away_runs:
            break

        # ---- bottom: home bats, faces away pitching ----
        a_pitch["inning"] = inning
        r, h_idx = _simulate_half(home_probs, h_idx, a_pitch, h_t,
                                  pitcher_is_sp=not a_pitch["bullpen_in"], rng=rng,
                                  ghost_on_second=ghost)
        home_runs += r
        if inning <= 5: f5_home += r
        if inning == 1: inning1_runs += r
        if first_to_score is None and r > 0: first_to_score = "H"

        if inning >= ctx.innings and home_runs != away_runs:
            break
        inning += 1
        if inning > 30:  # safety valve
            break

    return {
        "home_runs": home_runs,
        "away_runs": away_runs,
        "total": home_runs + away_runs,
        "home_win": int(home_runs > away_runs),
        "run_diff": home_runs - away_runs,
        "f5_total": f5_home + f5_away,
        "f5_home_win": int(f5_home > f5_away),
        "inning1_runs": inning1_runs,
        "nrfi": int(inning1_runs == 0),
        "first_to_score": first_to_score,
        "home_hits": h_t.hits, "home_tb": h_t.tb, "home_hr": h_t.hr,
        "away_hits": a_t.hits, "away_tb": a_t.tb, "away_hr": a_t.hr,
        # a_t.k_sp = K's the away lineup took vs the home starter = home starter's K's
        "home_starter_k": a_t.k_sp,
        "away_starter_k": h_t.k_sp,
    }


def run_simulation(home: Team, away: Team, ctx: Optional[GameContext] = None,
                   n_sims: int = config.N_SIMS_DEFAULT, seed: int = 0) -> dict:
    """Run N simulations and aggregate into calibrated-ish probabilities + prop dists."""
    ctx = ctx or GameContext()
    rng = np.random.default_rng(seed)

    home_wins = 0
    totals = np.empty(n_sims)
    diffs = np.empty(n_sims)
    nrfi = 0
    f5_home_wins = 0
    fts_home = 0
    home_starter_k = np.empty(n_sims)
    away_starter_k = np.empty(n_sims)
    home_hr = np.zeros(9); away_hr = np.zeros(9)
    home_hits = np.zeros(9); away_hits = np.zeros(9)
    home_tb = np.zeros(9); away_tb = np.zeros(9)
    home_hr_any = np.zeros(9); away_hr_any = np.zeros(9)
    home_2hit = np.zeros(9); away_2hit = np.zeros(9)

    for s in range(n_sims):
        g = simulate_game(home, away, ctx, rng)
        home_wins += g["home_win"]
        totals[s] = g["total"]
        diffs[s] = g["run_diff"]
        nrfi += g["nrfi"]
        f5_home_wins += g["f5_home_win"]
        fts_home += int(g["first_to_score"] == "H")
        home_starter_k[s] = g["home_starter_k"]
        away_starter_k[s] = g["away_starter_k"]
        home_hr += g["home_hr"]; away_hr += g["away_hr"]
        home_hits += g["home_hits"]; away_hits += g["away_hits"]
        home_tb += g["home_tb"]; away_tb += g["away_tb"]
        home_hr_any += (g["home_hr"] > 0); away_hr_any += (g["away_hr"] > 0)
        home_2hit += (g["home_hits"] >= 2); away_2hit += (g["away_hits"] >= 2)

    def over_grid(arr, lines):
        return {f"over_{ln}": float(np.mean(arr > ln)) for ln in lines}

    return {
        "n_sims": n_sims,
        "p_home_win": home_wins / n_sims,
        "p_away_win": 1 - home_wins / n_sims,
        "exp_total": float(totals.mean()),
        "total_over": over_grid(totals, [6.5, 7.5, 8.5, 9.5, 10.5]),
        "p_home_-1.5": float(np.mean(diffs > 1.5)),
        "p_home_+1.5": float(np.mean(diffs > -1.5)),
        "p_nrfi": nrfi / n_sims,
        "p_yrfi": 1 - nrfi / n_sims,
        "p_f5_home": f5_home_wins / n_sims,
        "p_first_to_score_home": fts_home / n_sims,
        "home_starter_k": {
            "mean": float(home_starter_k.mean()),
            "over": over_grid(home_starter_k, [4.5, 5.5, 6.5, 7.5, 8.5]),
        },
        "away_starter_k": {
            "mean": float(away_starter_k.mean()),
            "over": over_grid(away_starter_k, [4.5, 5.5, 6.5, 7.5, 8.5]),
        },
        "home_batters": [
            {"name": home.lineup[i].name,
             "exp_hits": float(home_hits[i] / n_sims),
             "exp_tb": float(home_tb[i] / n_sims),
             "exp_hr": float(home_hr[i] / n_sims),
             "p_hr": float(home_hr_any[i] / n_sims),
             "p_2plus_hits": float(home_2hit[i] / n_sims)}
            for i in range(9)
        ],
        "away_batters": [
            {"name": away.lineup[i].name,
             "exp_hits": float(away_hits[i] / n_sims),
             "exp_tb": float(away_tb[i] / n_sims),
             "exp_hr": float(away_hr[i] / n_sims),
             "p_hr": float(away_hr_any[i] / n_sims),
             "p_2plus_hits": float(away_2hit[i] / n_sims)}
            for i in range(9)
        ],
    }
