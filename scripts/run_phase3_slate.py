"""Generate and persist the complete Phase 3 output contract for an MLB slate."""
from __future__ import annotations

import argparse
import datetime as dt
import time

from athena_api.database import SessionLocal
from features.load_teams import build_game, load_rate_tables
from ingest.pull_mlb_statsapi import _date, snapshot_probables
from pipeline.phase3_outputs import PredictionContext, store_simulation
from sim.markov_game import run_simulation

TERMINAL_STATES = {
    "In Progress",
    "Final",
    "Game Over",
    "Completed",
    "Postponed",
    "Suspended",
    "Cancelled",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--sims", type=int, default=20_000)
    parser.add_argument("--confirmed-only", action="store_true")
    parser.add_argument("--include-started", action="store_true")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--runtime-budget-seconds", type=float, default=900.0)
    args = parser.parse_args()

    slate_date = _date(args.date)
    games = snapshot_probables(slate_date)
    tables = load_rate_tables(args.season)
    if args.max_games:
        games = games[: args.max_games]

    started = time.perf_counter()
    completed = skipped = outputs = 0
    with SessionLocal() as db:
        for game in games:
            if not args.include_started and game["status"] in TERMINAL_STATES:
                skipped += 1
                continue
            spec = f"{game['away']}@{game['home']}"
            home, away, context, info = build_game(
                spec,
                slate_date,
                args.season,
                tables=tables,
            )
            sources = info["lineup_source"]
            confirmed = (
                sources.get("home") == "confirmed"
                and sources.get("away") == "confirmed"
            )
            if args.confirmed_only and not confirmed:
                skipped += 1
                continue
            seed = int(game["gamePk"])
            simulation = run_simulation(
                home,
                away,
                context,
                n_sims=args.sims,
                seed=seed,
            )
            result = store_simulation(
                db,
                simulation,
                PredictionContext(
                    game_id=f"{spec}-{slate_date}",
                    mlb_game_pk=game["gamePk"],
                    home_team_id=game["home_team_id"],
                    away_team_id=game["away_team_id"],
                    data_snapshot_id=f"rates-{args.season}-{slate_date}",
                    first_pitch_at=dt.datetime.fromisoformat(
                        game["gameDate"].replace("Z", "+00:00")
                    ),
                    lineup_status="confirmed" if confirmed else "projected",
                    rate_source_version=str(args.season),
                    simulation_seed=seed,
                ),
            )
            completed += 1
            outputs += result["created"]
            print(f"[phase3] {spec}: {result['created']} created, {result['reused']} reused")

    elapsed = time.perf_counter() - started
    print(
        f"[phase3] completed={completed} skipped={skipped} outputs={outputs} "
        f"elapsed={elapsed:.1f}s budget={args.runtime_budget_seconds:.1f}s"
    )
    if elapsed > args.runtime_budget_seconds:
        raise SystemExit("Phase 3 slate exceeded the configured runtime budget")


if __name__ == "__main__":
    main()
