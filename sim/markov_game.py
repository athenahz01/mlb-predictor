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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

import config
from features.pa_probabilities import Batter, Pitcher, matchup_rates

if TYPE_CHECKING:
    from models.transitions import TransitionModel
    from models.workload import StarterWorkload

EVENTS = config.EVENTS


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class Team:
    code: str
    lineup: list[Batter]  # 9 batters, in order
    starter: Pitcher
    bullpen: Pitcher  # compatibility aggregate/fallback
    pitch_limit: int | None = None  # expected pitch count (None -> league default)
    starter_workload: StarterWorkload | None = None
    bullpen_tiers: tuple[Pitcher, ...] = field(default_factory=tuple)

    def __post_init__(self):
        assert len(self.lineup) == 9, "lineup must be 9 batters"


@dataclass
class GameContext:
    park_code: str = "_DEFAULT"
    ump_k_mult: float = 1.0
    innings: int = 9
    hfa_mult: float = 1.045  # small home-offense bump -> ~53% home win baseline
    env_hr: float = 1.0  # game-day weather HR multiplier (1.0 = neutral)
    env_hit: float = 1.0  # game-day weather hit multiplier
    transition_model: TransitionModel | None = None


# --------------------------------------------------------------------------
# Pre-compute per-(batter,pitcher) PA probability vectors so we sample fast.
# We build two matchup tables per team: vs the opposing starter and vs the
# opposing bullpen. Times-through-order is applied as a small additive bump.
# --------------------------------------------------------------------------
def _precompute(team: Team, opp_starter: Pitcher, opp_pens: list[Pitcher], ctx: GameContext):
    """Return matchup vectors for the starter and each bullpen tier."""
    table = {}
    for i, bat in enumerate(team.lineup):
        vs_sp = matchup_rates(
            bat,
            opp_starter,
            ctx.park_code,
            ctx.ump_k_mult,
            tto_bump=0.0,
            env_hr=ctx.env_hr,
            env_hit=ctx.env_hit,
        )
        vs_sp_tto = matchup_rates(
            bat,
            opp_starter,
            ctx.park_code,
            ctx.ump_k_mult,
            tto_bump=config.TTO_PENALTY,
            env_hr=ctx.env_hr,
            env_hit=ctx.env_hit,
        )
        table[i] = {
            "SP": np.array([vs_sp[e] for e in EVENTS]),
            "SP_tto": np.array([vs_sp_tto[e] for e in EVENTS]),
        }
        for pen_index, pen in enumerate(opp_pens):
            vs_bp = matchup_rates(
                bat,
                pen,
                ctx.park_code,
                ctx.ump_k_mult,
                tto_bump=0.0,
                env_hr=ctx.env_hr,
                env_hit=ctx.env_hit,
            )
            table[i][f"BP_{pen_index}"] = np.array([vs_bp[e] for e in EVENTS])
    return table


# event indices
iBB, iHBP, i1B, i2B, i3B, iHR, iK, iOUT = range(8)


def _keyed_uniform(seed: int, *coordinates: int) -> float:
    """Deterministic SplitMix64 draw that does not perturb the event RNG."""
    mask = (1 << 64) - 1
    value = int(seed) & mask
    for coordinate in coordinates:
        value = (value + 0x9E3779B97F4A7C15 + int(coordinate)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return (value & ((1 << 53) - 1)) / float(1 << 53)


@dataclass
class _Tally:
    runs: int = 0
    hits: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=int))
    tb: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=int))
    hr: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=int))
    k_sp: int = 0  # strikeouts by the starter
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
        return (0, 0, 1), b1 + b2 + b3  # batter to 3rd, all score
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


def _simulate_half(
    lineup_probs, lineup_start_idx, pitch_state, tally, pitcher_is_sp, rng, ghost_on_second=False
):
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
        pitcher_was_starter = not pitch_state["bullpen_in"]
        if pitcher_was_starter:
            key = "SP_tto" if times_through >= 2 else "SP"
        else:
            tier_count = pitch_state.get("bullpen_tier_count", 1)
            inning = pitch_state["inning"]
            if tier_count == 1:
                tier_index = 0
            elif inning >= 8:
                tier_index = tier_count - 1
            elif inning >= 6:
                tier_index = min(1, tier_count - 1)
            else:
                tier_index = 0
            key = f"BP_{tier_index}"
        cumulative = lineup_probs[idx][key]
        ev = min(7, int(np.searchsorted(cumulative, rng.random(), side="right")))
        slot_pa_number = int(pitch_state["slot_pa"][idx]) + 1
        pitch_state["slot_pa"][idx] = slot_pa_number
        playing_time = pitch_state["lineup"][idx].playing_time
        starter_batter_active = playing_time is None or _keyed_uniform(
            pitch_state["playing_time_seed"],
            idx,
            slot_pa_number,
            pitch_state["inning"],
        ) < playing_time.probability_active(slot_pa_number)

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
        if starter_batter_active and ev in (i1B, i2B, i3B, iHR):
            tally.hits[idx] += 1
            tally.tb[idx] += {i1B: 1, i2B: 2, i3B: 3, iHR: 4}[ev]
            if ev == iHR:
                tally.hr[idx] += 1

        outs_before = outs
        empirical = None
        if pitch_state.get("transition_model") is not None:
            empirical = pitch_state["transition_model"].sample(
                outs,
                (b1, b2, b3),
                EVENTS[int(ev)],
                rng,
            )
        if empirical is not None:
            b1, b2, b3 = empirical.bases
            runs += empirical.runs
            outs = min(3, outs + empirical.outs_added)
        elif ev == iK or ev == iOUT:
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
        if pitcher_was_starter:
            pitch_state["starter_outs"] += outs - outs_before

        idx = (idx + 1) % 9

        # bullpen hook (checked after each PA)
        if (
            not pitch_state["bullpen_in"]
            and pitcher_is_sp
            and (
                pitch_state["pitches"] >= pitch_state.get("limit", config.STARTER_PITCH_LIMIT)
                or pitch_state["inning"] >= config.STARTER_IP_SOFT_CAP
            )
        ):
            pitch_state["starter_pitches"] = pitch_state["pitches"]
            pitch_state["starter_batters_faced"] = pitch_state["batters_faced"]
            pitch_state["bullpen_in"] = True

    return runs, idx


def _game_probabilities(home: Team, away: Team, ctx: GameContext):
    """Precompute immutable PA probability tables once per simulation batch."""
    away_pens = list(away.bullpen_tiers) or [away.bullpen]
    home_pens = list(home.bullpen_tiers) or [home.bullpen]
    home_probs = _precompute(home, away.starter, away_pens, ctx)
    away_probs = _precompute(away, home.starter, home_pens, ctx)

    # home-field advantage: nudge home offensive events up slightly, renormalise
    if ctx.hfa_mult != 1.0:
        for i in home_probs:
            for key in home_probs[i]:
                v = home_probs[i][key].copy()
                for e_idx in (iBB, i1B, i2B, i3B, iHR):
                    v[e_idx] *= ctx.hfa_mult
                home_probs[i][key] = v / v.sum()
    for matchup_table in (home_probs, away_probs):
        for batter_matchups in matchup_table.values():
            for key, probabilities in batter_matchups.items():
                cumulative = np.cumsum(probabilities)
                cumulative[-1] = 1.0
                batter_matchups[key] = cumulative
    return home_probs, away_probs, home_pens, away_pens


def simulate_game(
    home: Team,
    away: Team,
    ctx: GameContext,
    rng,
    *,
    playing_time_seed: int = 0,
    probabilities=None,
) -> dict:
    """Simulate ONE game. Returns a dict of outcomes for this single realisation."""
    if probabilities is None:
        probabilities = _game_probabilities(home, away, ctx)
    home_probs, away_probs, home_pens, away_pens = probabilities

    h_t = _Tally(hits=np.zeros(9, int), tb=np.zeros(9, int), hr=np.zeros(9, int))
    a_t = _Tally(hits=np.zeros(9, int), tb=np.zeros(9, int), hr=np.zeros(9, int))

    h_limit = (
        home.starter_workload.sample_pitch_limit(rng)
        if home.starter_workload
        else home.pitch_limit or config.STARTER_PITCH_LIMIT
    )
    a_limit = (
        away.starter_workload.sample_pitch_limit(rng)
        if away.starter_workload
        else away.pitch_limit or config.STARTER_PITCH_LIMIT
    )
    h_pitch: dict[str, Any] = {
        "pitches": 0,
        "batters_faced": 0,
        "bullpen_in": False,
        "inning": 0,
        "limit": h_limit,
        "starter_outs": 0,
        "bullpen_tier_count": len(home_pens),
        "slot_pa": np.zeros(9, dtype=int),
        "lineup": away.lineup,
        "playing_time_seed": playing_time_seed,
        "transition_model": ctx.transition_model,
    }
    a_pitch: dict[str, Any] = {
        "pitches": 0,
        "batters_faced": 0,
        "bullpen_in": False,
        "inning": 0,
        "limit": a_limit,
        "starter_outs": 0,
        "bullpen_tier_count": len(away_pens),
        "slot_pa": np.zeros(9, dtype=int),
        "lineup": home.lineup,
        "playing_time_seed": playing_time_seed ^ 0x5DEECE66D,
        "transition_model": ctx.transition_model,
    }

    h_idx = a_idx = 0
    home_runs = away_runs = 0
    f5_home = f5_away = 0  # first-five-innings runs
    inning1_runs = 0
    first_to_score = None  # 'H' / 'A'

    inning = 1
    while True:
        extras = inning > ctx.innings
        ghost = config.GHOST_RUNNER_EXTRAS and extras

        # ---- top: away bats, faces home pitching ----
        h_pitch["inning"] = inning
        r, a_idx = _simulate_half(
            away_probs,
            a_idx,
            h_pitch,
            a_t,
            pitcher_is_sp=not h_pitch["bullpen_in"],
            rng=rng,
            ghost_on_second=ghost,
        )
        away_runs += r
        if inning <= 5:
            f5_away += r
        if inning == 1:
            inning1_runs += r
        if first_to_score is None and r > 0:
            first_to_score = "A"

        # walk-off short-circuit: home already leads, skip bottom of 9th+
        if inning >= ctx.innings and home_runs > away_runs:
            break

        # ---- bottom: home bats, faces away pitching ----
        a_pitch["inning"] = inning
        r, h_idx = _simulate_half(
            home_probs,
            h_idx,
            a_pitch,
            h_t,
            pitcher_is_sp=not a_pitch["bullpen_in"],
            rng=rng,
            ghost_on_second=ghost,
        )
        home_runs += r
        if inning <= 5:
            f5_home += r
        if inning == 1:
            inning1_runs += r
        if first_to_score is None and r > 0:
            first_to_score = "H"

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
        "home_hits": h_t.hits,
        "home_tb": h_t.tb,
        "home_hr": h_t.hr,
        "away_hits": a_t.hits,
        "away_tb": a_t.tb,
        "away_hr": a_t.hr,
        # a_t.k_sp = K's the away lineup took vs the home starter = home starter's K's
        "home_starter_k": a_t.k_sp,
        "away_starter_k": h_t.k_sp,
        "home_starter_pitches": h_pitch.get("starter_pitches", h_pitch["pitches"]),
        "away_starter_pitches": a_pitch.get("starter_pitches", a_pitch["pitches"]),
        "home_starter_batters_faced": h_pitch.get(
            "starter_batters_faced", h_pitch["batters_faced"]
        ),
        "away_starter_batters_faced": a_pitch.get(
            "starter_batters_faced", a_pitch["batters_faced"]
        ),
        "home_starter_innings": h_pitch["starter_outs"] / 3.0,
        "away_starter_innings": a_pitch["starter_outs"] / 3.0,
        "innings_played": inning,
    }


def run_simulation(
    home: Team,
    away: Team,
    ctx: GameContext | None = None,
    n_sims: int = config.N_SIMS_DEFAULT,
    seed: int = 0,
) -> dict:
    """Run N simulations and aggregate into calibrated-ish probabilities + prop dists."""
    ctx = ctx or GameContext()
    rng = np.random.default_rng(seed)
    probabilities = _game_probabilities(home, away, ctx)

    home_wins = 0
    totals = np.empty(n_sims)
    diffs = np.empty(n_sims)
    nrfi = 0
    f5_home_wins = 0
    fts_home = 0
    home_starter_k = np.empty(n_sims)
    away_starter_k = np.empty(n_sims)
    home_hr = np.zeros(9)
    away_hr = np.zeros(9)
    home_hits = np.zeros(9)
    away_hits = np.zeros(9)
    home_tb = np.zeros(9)
    away_tb = np.zeros(9)
    home_hr_any = np.zeros(9)
    away_hr_any = np.zeros(9)
    home_hit_any = np.zeros(9)
    away_hit_any = np.zeros(9)
    home_2hit = np.zeros(9)
    away_2hit = np.zeros(9)
    home_runs = np.empty(n_sims)
    away_runs = np.empty(n_sims)
    innings_played = np.empty(n_sims)
    home_starter_pitches = np.empty(n_sims)
    away_starter_pitches = np.empty(n_sims)
    home_starter_bf = np.empty(n_sims)
    away_starter_bf = np.empty(n_sims)
    home_starter_ip = np.empty(n_sims)
    away_starter_ip = np.empty(n_sims)

    for s in range(n_sims):
        g = simulate_game(
            home,
            away,
            ctx,
            rng,
            playing_time_seed=seed * 1_000_003 + s,
            probabilities=probabilities,
        )
        home_wins += g["home_win"]
        home_runs[s] = g["home_runs"]
        away_runs[s] = g["away_runs"]
        totals[s] = g["total"]
        diffs[s] = g["run_diff"]
        nrfi += g["nrfi"]
        f5_home_wins += g["f5_home_win"]
        fts_home += int(g["first_to_score"] == "H")
        home_starter_k[s] = g["home_starter_k"]
        away_starter_k[s] = g["away_starter_k"]
        home_hr += g["home_hr"]
        away_hr += g["away_hr"]
        home_hits += g["home_hits"]
        away_hits += g["away_hits"]
        home_tb += g["home_tb"]
        away_tb += g["away_tb"]
        home_hr_any += g["home_hr"] > 0
        away_hr_any += g["away_hr"] > 0
        home_hit_any += g["home_hits"] > 0
        away_hit_any += g["away_hits"] > 0
        home_2hit += g["home_hits"] >= 2
        away_2hit += g["away_hits"] >= 2
        innings_played[s] = g["innings_played"]
        home_starter_pitches[s] = g["home_starter_pitches"]
        away_starter_pitches[s] = g["away_starter_pitches"]
        home_starter_bf[s] = g["home_starter_batters_faced"]
        away_starter_bf[s] = g["away_starter_batters_faced"]
        home_starter_ip[s] = g["home_starter_innings"]
        away_starter_ip[s] = g["away_starter_innings"]

    def over_grid(arr, lines):
        return {f"over_{ln}": float(np.mean(arr > ln)) for ln in lines}

    return {
        "n_sims": n_sims,
        "p_home_win": home_wins / n_sims,
        "p_away_win": 1 - home_wins / n_sims,
        "exp_total": float(totals.mean()),
        "exp_home_runs": float(home_runs.mean()),
        "exp_away_runs": float(away_runs.mean()),
        "home_run_distribution": {
            str(int(value)): float(np.mean(home_runs == value)) for value in np.unique(home_runs)
        },
        "away_run_distribution": {
            str(int(value)): float(np.mean(away_runs == value)) for value in np.unique(away_runs)
        },
        "total_run_distribution": {
            str(int(value)): float(np.mean(totals == value)) for value in np.unique(totals)
        },
        "total_over": over_grid(totals, [6.5, 7.5, 8.5, 9.5, 10.5]),
        "home_team_total_over": over_grid(home_runs, [2.5, 3.5, 4.5, 5.5]),
        "away_team_total_over": over_grid(away_runs, [2.5, 3.5, 4.5, 5.5]),
        "p_home_shutout": float(np.mean(home_runs == 0)),
        "p_away_shutout": float(np.mean(away_runs == 0)),
        "p_home_5plus_runs": float(np.mean(home_runs >= 5)),
        "p_away_5plus_runs": float(np.mean(away_runs >= 5)),
        "p_extra_innings": float(np.mean(innings_played > ctx.innings)),
        "p_home_-1.5": float(np.mean(diffs > 1.5)),
        "p_home_+1.5": float(np.mean(diffs > -1.5)),
        "p_nrfi": nrfi / n_sims,
        "p_yrfi": 1 - nrfi / n_sims,
        "p_f5_home": f5_home_wins / n_sims,
        "p_first_to_score_home": fts_home / n_sims,
        "home_starter_k": {
            "player_id": home.starter.mlb_id,
            "name": home.starter.name,
            "mean": float(home_starter_k.mean()),
            "over": over_grid(home_starter_k, [4.5, 5.5, 6.5, 7.5, 8.5]),
            "distribution": {
                str(int(value)): float(np.mean(home_starter_k == value))
                for value in np.unique(home_starter_k)
            },
            "expected_pitches": float(home_starter_pitches.mean()),
            "expected_batters_faced": float(home_starter_bf.mean()),
            "expected_innings": float(home_starter_ip.mean()),
            "probability_starting_inning": {
                str(inning): (
                    home.starter_workload.probability_starting_inning(inning)
                    if home.starter_workload
                    else float(np.mean(home_starter_ip >= inning - 1))
                )
                for inning in range(2, 8)
            },
            "data_quality_flags": list(home.starter.data_quality_flags),
        },
        "away_starter_k": {
            "player_id": away.starter.mlb_id,
            "name": away.starter.name,
            "mean": float(away_starter_k.mean()),
            "over": over_grid(away_starter_k, [4.5, 5.5, 6.5, 7.5, 8.5]),
            "distribution": {
                str(int(value)): float(np.mean(away_starter_k == value))
                for value in np.unique(away_starter_k)
            },
            "expected_pitches": float(away_starter_pitches.mean()),
            "expected_batters_faced": float(away_starter_bf.mean()),
            "expected_innings": float(away_starter_ip.mean()),
            "probability_starting_inning": {
                str(inning): (
                    away.starter_workload.probability_starting_inning(inning)
                    if away.starter_workload
                    else float(np.mean(away_starter_ip >= inning - 1))
                )
                for inning in range(2, 8)
            },
            "data_quality_flags": list(away.starter.data_quality_flags),
        },
        "home_batters": [
            {
                "name": home.lineup[i].name,
                "player_id": home.lineup[i].mlb_id,
                "exp_hits": float(home_hits[i] / n_sims),
                "exp_tb": float(home_tb[i] / n_sims),
                "exp_hr": float(home_hr[i] / n_sims),
                "p_hit": float(home_hit_any[i] / n_sims),
                "p_hr": float(home_hr_any[i] / n_sims),
                "p_2plus_hits": float(home_2hit[i] / n_sims),
                "data_quality_flags": list(home.lineup[i].data_quality_flags),
            }
            for i in range(9)
        ],
        "away_batters": [
            {
                "name": away.lineup[i].name,
                "player_id": away.lineup[i].mlb_id,
                "exp_hits": float(away_hits[i] / n_sims),
                "exp_tb": float(away_tb[i] / n_sims),
                "exp_hr": float(away_hr[i] / n_sims),
                "p_hit": float(away_hit_any[i] / n_sims),
                "p_hr": float(away_hr_any[i] / n_sims),
                "p_2plus_hits": float(away_2hit[i] / n_sims),
                "data_quality_flags": list(away.lineup[i].data_quality_flags),
            }
            for i in range(9)
        ],
        "data_quality_flags": sorted(
            {
                *home.starter.data_quality_flags,
                *away.starter.data_quality_flags,
                *home.bullpen.data_quality_flags,
                *away.bullpen.data_quality_flags,
                *(flag for pen in home.bullpen_tiers for flag in pen.data_quality_flags),
                *(flag for pen in away.bullpen_tiers for flag in pen.data_quality_flags),
                *(flag for batter in home.lineup for flag in batter.data_quality_flags),
                *(flag for batter in away.lineup for flag in batter.data_quality_flags),
            }
        ),
    }
