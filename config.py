"""
Central configuration for the MLB predictor.

Everything that is a tunable constant or a league baseline lives here so the
rest of the pipeline reads from one place. Park factors and league rates are
the two things you will re-fit each season; keep them version-pinned.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (Windows-friendly: everything is relative to this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAPSHOTS = DATA / "snapshots"      # frozen per-day pulls (pin these, never auto-overwrite)
CACHE = DATA / "cache"              # pybaseball cache / scratch
ARTIFACTS = ROOT / "models" / "artifacts"   # saved model artifacts (shipped only)
LEDGER_PATH = ROOT / "ledger" / "ledger.json"

for _p in (DATA, SNAPSHOTS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# League-average per-PA outcome rates (the log5 anchor `l`)
# 2023+ regime (post pitch-clock / shift-restriction). Re-fit each spring.
# Must sum to 1.0. Order is the canonical event order used everywhere.
# ---------------------------------------------------------------------------
EVENTS = ["BB", "HBP", "1B", "2B", "3B", "HR", "K", "IP_OUT"]

LEAGUE_PA_RATES = {
    "BB":     0.085,
    "HBP":    0.011,
    "1B":     0.140,
    "2B":     0.045,
    "3B":     0.004,
    "HR":     0.033,
    "K":      0.224,
    "IP_OUT": 0.458,   # in-play, non-K, non-hit outs (implied BABIP ~ .292)
}
assert abs(sum(LEAGUE_PA_RATES.values()) - 1.0) < 1e-9, "league rates must sum to 1"

# ---------------------------------------------------------------------------
# Park factors, indexed to 100 (=neutral). Handedness-controlled HR factors.
# Only a few parks shown; fill from Baseball Savant statcast-park-factors.
# `hr` multiplies HR rate, `hit` multiplies non-HR hit (1B/2B/3B) rate.
# ---------------------------------------------------------------------------
PARK_FACTORS = {
    # approximate multipliers vs league-average (hr = home-run factor, hit = base-hit factor).
    # Refine later with measured Statcast/FanGraphs park factors; these are reasonable priors.
    "COL": {"hr": 1.18, "hit": 1.10},   # Coors - altitude outlier
    "CIN": {"hr": 1.27, "hit": 1.03},   # Great American - HR bandbox
    "NYY": {"hr": 1.12, "hit": 1.00},   # short porch
    "PHI": {"hr": 1.10, "hit": 1.02},   # Citizens Bank
    "MIL": {"hr": 1.08, "hit": 1.00},   # American Family
    "CWS": {"hr": 1.07, "hit": 0.99},   # Rate Field
    "LAD": {"hr": 1.06, "hit": 0.98},   # Dodger Stadium
    "TEX": {"hr": 1.05, "hit": 1.02},   # Globe Life
    "HOU": {"hr": 1.04, "hit": 1.01},   # Daikin Park
    "ARI": {"hr": 1.04, "hit": 1.03},   # Chase Field
    "ATL": {"hr": 1.02, "hit": 1.00},   # Truist
    "CHC": {"hr": 1.02, "hit": 1.01},   # Wrigley - wind dependent
    "BAL": {"hr": 1.02, "hit": 1.00},   # Camden (post wall move)
    "TOR": {"hr": 1.02, "hit": 1.00},   # Rogers Centre
    "WSH": {"hr": 1.01, "hit": 1.00},   # Nationals Park
    "LAA": {"hr": 1.00, "hit": 0.99},   # Angel Stadium
    "ATH": {"hr": 1.00, "hit": 1.00},   # Sutter Health Park (new, neutral prior)
    "MIN": {"hr": 0.99, "hit": 1.00},   # Target Field
    "BOS": {"hr": 0.97, "hit": 1.07},   # Fenway - hits up, HR ~neutral
    "TB":  {"hr": 0.97, "hit": 0.97},   # Tropicana
    "CLE": {"hr": 0.96, "hit": 0.98},   # Progressive
    "NYM": {"hr": 0.95, "hit": 0.98},   # Citi Field
    "STL": {"hr": 0.94, "hit": 1.00},   # Busch
    "DET": {"hr": 0.94, "hit": 0.99},   # Comerica
    "SEA": {"hr": 0.93, "hit": 0.95},   # T-Mobile - suppresses
    "SD":  {"hr": 0.92, "hit": 0.97},   # Petco - suppresses
    "KC":  {"hr": 0.92, "hit": 1.02},   # Kauffman - big OF, doubles/triples
    "PIT": {"hr": 0.92, "hit": 1.00},   # PNC
    "SF":  {"hr": 0.90, "hit": 0.97},   # Oracle - marine layer
    "MIA": {"hr": 0.86, "hit": 0.96},   # loanDepot - biggest suppressor
    "_DEFAULT": {"hr": 1.00, "hit": 1.00},
}

def park(code: str) -> dict:
    return PARK_FACTORS.get(code, PARK_FACTORS["_DEFAULT"])

# ---------------------------------------------------------------------------
# Simulation knobs
# ---------------------------------------------------------------------------
N_SIMS_DEFAULT = 20_000
STARTER_PITCH_LIMIT = 95            # pitches before the bullpen is summoned
STARTER_IP_SOFT_CAP = 6.0          # also pull after this many innings
TTO_PENALTY = 0.010                # per-PA additive bump to opp offense each time through order (3rd+)
GHOST_RUNNER_EXTRAS = True         # Manfred runner on 2nd in the 10th+

# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = {
    "winner": "KXMLBGAME",
    "total":  "KXMLBTOTAL",
    "ws":     "KXMLB",
    "wins":   "KXMLBWINS",
}

# MLB Stats API (official, no key)
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
