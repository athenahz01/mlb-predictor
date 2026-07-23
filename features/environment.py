"""
features/environment.py
-----------------------
Game-day context layer: everything about TODAY that the rate tables can't know.

  weather_mults(park, date)      -> {"hr": x, "hit": y} from game-time temp + wind
  ump_k_mult(game_pk)            -> plate umpire strikeout multiplier (if known)
  starter_pitch_limit(pid)       -> expected pitch count from recent starts
  unavailable_relievers(team_id, date) -> {pid: weight_mult} from recent usage

All fetchers fail SOFT: any error returns the neutral value, so a dead API can
never block the slate. Weather uses Open-Meteo (free, no key). Umpire factors
come from data/snapshots/ump_factors.json (see backtest/build_ump_factors.py).

Effect sizes are conservative priors from the public research:
  - temperature: ball carries ~4-6 ft further per +40F -> ~0.5%/F on HR odds
  - wind: out/in at 15+ mph moves totals ~0.5-1.5 runs -> ~1.2%/mph on HR, capped
  - umpire: zones move K rates several percent; factors are regressed at K=60 games
"""

from __future__ import annotations

import datetime as dt
import json

import requests

import config
from models.workload import StarterWorkload, fit_starter_workload

S = requests.Session()

# lat, lon, approximate center-field bearing (deg from north), roof?
PARKS = {
    "ARI": (33.445, -112.067, 0, True),
    "AZ": (33.445, -112.067, 0, True),
    "ATL": (33.891, -84.468, 40, False),
    "BAL": (39.284, -76.622, 30, False),
    "BOS": (42.346, -71.097, 55, False),
    "CHC": (41.948, -87.656, 40, False),
    "CWS": (41.830, -87.634, 125, False),
    "CIN": (39.097, -84.507, 120, False),
    "CLE": (41.496, -81.685, 0, False),
    "COL": (39.756, -104.994, 5, False),
    "DET": (42.339, -83.049, 150, False),
    "HOU": (29.757, -95.356, 345, True),
    "KC": (39.051, -94.480, 45, False),
    "LAA": (33.800, -117.883, 65, False),
    "LAD": (34.074, -118.240, 25, False),
    "MIA": (25.778, -80.220, 40, True),
    "MIL": (43.028, -87.971, 130, True),
    "MIN": (44.982, -93.278, 90, False),
    "NYM": (40.757, -73.846, 15, False),
    "NYY": (40.829, -73.926, 55, False),
    "ATH": (38.580, -121.513, 60, False),
    "PHI": (39.906, -75.166, 10, False),
    "PIT": (40.447, -80.006, 115, False),
    "SD": (32.707, -117.157, 0, False),
    "SEA": (47.591, -122.332, 45, True),
    "SF": (37.778, -122.389, 90, False),
    "STL": (38.623, -90.193, 60, False),
    "TB": (27.768, -82.653, 45, True),
    "TEX": (32.747, -97.084, 135, True),
    "TOR": (43.641, -79.389, 345, True),
    "WSH": (38.873, -77.007, 25, False),
}

TEMP_HR_PER_F = 0.005  # HR odds multiplier slope per degree F vs 70F
WIND_HR_PER_MPH = 0.012  # HR multiplier slope per mph of out/in component
ENV_CLAMP = (0.80, 1.25)


def _air_density(temp_f: float, humidity_pct: float, pressure_hpa: float) -> float:
    """Moist-air density in kg/m^3 using forecast pressure and humidity."""
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    temp_k = temp_c + 273.15
    saturation_hpa = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3))
    vapor_pa = humidity_pct / 100.0 * saturation_hpa * 100.0
    pressure_pa = pressure_hpa * 100.0
    dry_pa = max(0.0, pressure_pa - vapor_pa)
    return dry_pa / (287.05 * temp_k) + vapor_pa / (461.495 * temp_k)


def weather_mults(park: str, date: str, hour_utc: int = 23) -> dict:
    """HR/hit multipliers from temperature, wind, humidity, and air density."""
    neutral = {"hr": 1.0, "hit": 1.0, "detail": "neutral"}
    p = PARKS.get(park)
    if not p:
        return neutral
    lat, lon, cf_bearing, roof = p
    if roof:
        return {"hr": 1.0, "hit": 1.0, "detail": "roof"}
    try:
        r = S.get(
            "https://api.open-meteo.com/v1/forecast",
            timeout=15,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": (
                    "temperature_2m,relative_humidity_2m,surface_pressure,"
                    "wind_speed_10m,wind_direction_10m"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "start_date": date,
                "end_date": date,
            },
        ).json()
        hh = r["hourly"]
        idx = min(hour_utc, len(hh["temperature_2m"]) - 1)
        temp = hh["temperature_2m"][idx]
        wspd = hh["wind_speed_10m"][idx]
        wdir = hh["wind_direction_10m"][idx]  # direction wind comes FROM
        humidity = hh["relative_humidity_2m"][idx]
        pressure = hh["surface_pressure"][idx]
    except Exception:
        return neutral
    import math

    hr = 1.0 + TEMP_HR_PER_F * (temp - 70.0)
    # out-component: wind blowing TOWARD center field = (wdir - cf_bearing) ~ 180
    out_comp = -math.cos(math.radians(wdir - cf_bearing)) * wspd
    hr *= 1.0 + WIND_HR_PER_MPH * out_comp
    density = _air_density(temp, humidity, pressure)
    reference_density = _air_density(70.0, 50.0, 1013.25)
    hr *= (reference_density / density) ** 0.60
    hr = max(ENV_CLAMP[0], min(ENV_CLAMP[1], hr))
    hit = max(0.95, min(1.05, 1.0 + 0.001 * (temp - 70.0)))
    return {
        "hr": round(hr, 3),
        "hit": round(hit, 3),
        "detail": (
            f"{temp:.0f}F {humidity:.0f}% RH density {density:.3f}kg/m3 "
            f"wind {wspd:.0f}mph out-comp {out_comp:+.0f}"
        ),
    }


def ump_k_mult(game_pk: int) -> float:
    """Plate umpire K multiplier if the assignment + factors are known, else 1.0."""
    try:
        factors = json.loads((config.SNAPSHOTS / "ump_factors.json").read_text())
    except Exception:
        return 1.0
    try:
        r = S.get(f"{config.STATSAPI_BASE}.1/game/{game_pk}/feed/live", timeout=15).json()
        for off in r["liveData"]["boxscore"]["officials"]:
            if off.get("officialType") == "Home Plate":
                name = off["official"]["fullName"]
                return float(factors.get(name, {}).get("k_mult", 1.0))
    except Exception:
        pass
    return 1.0


def starter_pitch_limit(pid: int, season: int | None = None) -> int | None:
    """Expected pitch count: median of the last 3 starts' pitch counts, clamped."""
    season = season or dt.date.today().year
    try:
        r = S.get(
            f"{config.STATSAPI_BASE}/people/{pid}/stats",
            timeout=15,
            params={"stats": "gameLog", "group": "pitching", "season": season},
        ).json()
        counts = []
        for sp in r["stats"][0]["splits"]:
            n = sp.get("stat", {}).get("numberOfPitches")
            if n:
                counts.append(int(n))
        recent = counts[-3:]
        if not recent:
            return None
        recent.sort()
        med = recent[len(recent) // 2]
        return max(60, min(115, med))
    except Exception:
        return None


def starter_workload(
    pid: int,
    *,
    as_of_date: str | None = None,
    season: int | None = None,
    injury_return: bool = False,
    role: str = "starter",
) -> StarterWorkload:
    """Fit the workload challenger from game logs known before ``as_of_date``."""
    season = season or (
        dt.date.fromisoformat(as_of_date).year if as_of_date else dt.date.today().year
    )
    counts: list[int] = []
    dates: list[dt.date] = []
    try:
        response = S.get(
            f"{config.STATSAPI_BASE}/people/{pid}/stats",
            timeout=15,
            params={"stats": "gameLog", "group": "pitching", "season": season},
        ).json()
        cutoff = dt.date.fromisoformat(as_of_date) if as_of_date else None
        for split in response["stats"][0]["splits"]:
            game_date_raw = split.get("date")
            game_date = dt.date.fromisoformat(game_date_raw[:10]) if game_date_raw else None
            if cutoff and game_date and game_date >= cutoff:
                continue
            pitches = split.get("stat", {}).get("numberOfPitches")
            if pitches:
                counts.append(int(pitches))
                if game_date:
                    dates.append(game_date)
    except Exception:
        pass
    days_rest = None
    if dates and as_of_date:
        days_rest = (dt.date.fromisoformat(as_of_date) - dates[-1]).days
    return fit_starter_workload(
        counts,
        days_rest=days_rest,
        season_high=max(counts) if counts else None,
        injury_return=injury_return,
        role=role,
    )


def unavailable_relievers(team_id: int, date: str) -> dict:
    """{pitcher_id: weight_mult} from the previous two days' usage.
    Pitched yesterday -> 0.25 weight; two days ago -> 0.60; both -> 0.10."""
    weights: dict[int, float] = {}
    d = dt.date.fromisoformat(date)
    for back, w in ((1, 0.25), (2, 0.60)):
        day = (d - dt.timedelta(days=back)).isoformat()
        try:
            sched = S.get(
                f"{config.STATSAPI_BASE}/schedule",
                timeout=15,
                params={"sportId": 1, "date": day, "teamId": team_id},
            ).json()
            for dd in sched.get("dates", []):
                for g in dd.get("games", []):
                    box = S.get(
                        f"{config.STATSAPI_BASE}/game/{g['gamePk']}/boxscore", timeout=15
                    ).json()
                    for side in ("home", "away"):
                        t = box["teams"][side]
                        if t.get("team", {}).get("id") != team_id:
                            continue
                        pitchers = t.get("pitchers", [])
                        for pid in pitchers[1:]:  # skip that day's starter
                            weights[pid] = min(weights.get(pid, 1.0), w)
        except Exception:
            continue
    return weights
