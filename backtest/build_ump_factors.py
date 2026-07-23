"""
backtest/build_ump_factors.py
-----------------------------
Build plate-umpire strikeout tendencies from completed games. For every game in
results_{season}.json, fetch the boxscore + officials, credit the game's total
strikeouts to the home-plate umpire, and express each umpire's K rate as a
multiplier vs league, regressed toward 1.0 with K_GAMES=60 pseudo-games (umpire
effects are real but small; regression keeps small samples honest).

Writes data/snapshots/ump_factors.json, which features/environment.ump_k_mult
reads at slate time. Re-run weekly to keep factors current.

  python -m backtest.build_ump_factors --season 2026
"""
from __future__ import annotations

import argparse
import json

import config
from backtest.props_backtest import SESSION

K_GAMES = 60


def build(season: int, limit: int | None = None):
    results = json.loads((config.SNAPSHOTS / f"results_{season}.json").read_text())
    results = [g for g in results if g.get("home_score") is not None]
    if limit:
        results = results[:limit]

    per_ump: dict[str, list[int]] = {}
    done = 0
    for g in results:
        pk = g["gamePk"]
        try:
            live = SESSION.get(
                f"{config.STATSAPI_BASE}.1/game/{pk}/feed/live", timeout=20).json()
            officials = live["liveData"]["boxscore"]["officials"]
            hp = next(o["official"]["fullName"] for o in officials
                      if o.get("officialType") == "Home Plate")
            box = live["liveData"]["boxscore"]["teams"]
            ks = (box["home"]["teamStats"]["pitching"]["strikeOuts"]
                  + box["away"]["teamStats"]["pitching"]["strikeOuts"])
        except Exception:
            continue
        per_ump.setdefault(hp, []).append(ks)
        done += 1
        if done % 50 == 0:
            print(f"  ...{done} games")

    league = sum(sum(v) for v in per_ump.values()) / max(
        1, sum(len(v) for v in per_ump.values()))
    out = {}
    for ump, ks in sorted(per_ump.items()):
        n = len(ks)
        raw = (sum(ks) / n) / league
        k_mult = (n * raw + K_GAMES * 1.0) / (n + K_GAMES)
        out[ump] = {"games": n, "raw_k_per_game": round(sum(ks) / n, 2),
                    "k_mult": round(k_mult, 4)}
    path = config.SNAPSHOTS / "ump_factors.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[ump] {done} games, {len(out)} umpires (league {league:.1f} K/gm) -> {path.name}")
    ranked = sorted(out.items(), key=lambda kv: -kv[1]["k_mult"])
    for name, d in ranked[:3] + ranked[-3:]:
        print(f"  {name}: {d['games']} gms, k_mult {d['k_mult']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    build(ap.parse_args().season, ap.parse_args().limit)
