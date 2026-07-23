from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from athena_api.database import Base, SessionLocal, engine
from athena_api.ledger_service import create_revision, resolve_prediction
from athena_api.models import Prediction
from athena_api.schemas import PredictionCreate
from athena_api.settings import get_settings

MARKET_MAP = {
    "moneyline_home": ("game", "home_win_probability"),
    "total_over_8.5": ("game", "total_over_8_5"),
    "nrfi": ("game", "nrfi"),
    "home_sp_k_over_5.5": ("pitcher", "home_starter_strikeouts_over_5_5"),
    "away_sp_k_over_5.5": ("pitcher", "away_starter_strikeouts_over_5_5"),
    "batter_hr": ("batter", "home_run_probability"),
}


def parse_timestamp(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.UTC)
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def legacy_payload(row: dict[str, Any]) -> PredictionCreate:
    market = row.get("market", "unknown")
    category, statistic = MARKET_MAP.get(market, ("game", market.replace(".", "_")))
    meta = row.get("meta") or {}
    lineup_source = meta.get("lineup_source", "unknown")
    if isinstance(lineup_source, dict):
        states = {str(value) for value in lineup_source.values()}
        lineup_status = states.pop() if len(states) == 1 else "mixed"
    else:
        lineup_status = str(lineup_source or "unknown")
    flags = ["legacy_import", "missing_model_version", "missing_data_snapshot"]
    if category in {"pitcher", "batter"}:
        flags.append("missing_player_id")
    return PredictionCreate(
        game_id=str(row.get("game_id")),
        category=category,
        statistic=statistic,
        probability=row.get("model_p"),
        model_version="legacy-unknown",
        git_commit_sha=None,
        data_snapshot_id="legacy-unknown",
        rate_source_version="legacy-unknown",
        feature_version="legacy-unknown",
        simulation_settings={},
        simulation_seed=None,
        simulation_seed_policy="legacy-unknown",
        lineup_player_ids={},
        lineup_status=lineup_status,
        starting_pitcher_ids={},
        data_quality_flags=flags,
        evidence={
            "legacy_meta": meta,
            "market_probability": row.get("market_p"),
            "market_close_probability": row.get("market_close_p"),
            "main_reason": "Imported from the historical JSON ledger; detailed drivers were not stored.",
            "main_uncertainty": "Legacy model and data versions cannot be reconstructed.",
        },
        confidence="low",
        validation_status="provisional",
        revision_reason="historical_import",
        created_at=parse_timestamp(row.get("ts")),
    )


def inferred_result(row: Prediction, dashboard: dict[str, Any]) -> dict[str, Any] | None:
    if dashboard.get("status") != "final":
        return None
    home_score = dashboard.get("home_score")
    away_score = dashboard.get("away_score")
    if home_score is None or away_score is None:
        return None
    if row.statistic == "home_win_probability":
        return {"value": int(home_score > away_score), "source": "legacy_dashboard"}
    if row.statistic == "total_over_8_5":
        return {
            "value": int((home_score + away_score) > 8.5),
            "total_runs": home_score + away_score,
            "source": "legacy_dashboard",
        }
    return None


def migrate(
    source: Path,
    dashboard_source: Path,
    report_json: Path,
    report_md: Path,
) -> dict[str, Any]:
    Base.metadata.create_all(engine)
    rows = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    dashboard_rows = (
        json.loads(dashboard_source.read_text(encoding="utf-8"))
        if dashboard_source.exists()
        else []
    )
    dashboard_by_game = {
        f"{row.get('away')}@{row.get('home')}-{row.get('date')}": row
        for row in dashboard_rows
    }
    duplicate_keys = Counter(
        (row.get("game_id"), row.get("market"), row.get("ts")) for row in rows
    )
    imported = 0
    idempotent = 0
    resolved = 0
    unresolvable = 0
    missing_player_ids = 0
    missing_versions = 0
    with SessionLocal() as db:
        for row in rows:
            payload = legacy_payload(row)
            prediction, created = create_revision(db, payload)
            imported += int(created)
            idempotent += int(not created)
            missing_player_ids += int("missing_player_id" in payload.data_quality_flags)
            missing_versions += int("missing_model_version" in payload.data_quality_flags)
            result = inferred_result(prediction, dashboard_by_game.get(prediction.game_id, {}))
            if result is not None:
                resolve_prediction(db, prediction, result)
                resolved += 1
            elif prediction.final_result is None:
                unresolvable += 1
        database_rows = db.scalar(select(func.count(Prediction.id))) or 0

    source_games = {str(row.get("game_id")) for row in rows}
    dashboard_games = set(dashboard_by_game)
    mismatched = sorted(source_games.symmetric_difference(dashboard_games))
    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "existing_ledger_rows": len(rows),
        "dashboard_game_rows": len(dashboard_rows),
        "imported_rows": imported,
        "idempotent_existing_rows": idempotent,
        "database_rows": database_rows,
        "duplicate_rows": sum(count - 1 for count in duplicate_keys.values() if count > 1),
        "mismatched_games": len(mismatched),
        "mismatched_game_ids": mismatched,
        "unresolvable_rows": unresolvable,
        "resolved_from_dashboard": resolved,
        "missing_outcomes": unresolvable,
        "missing_player_ids": missing_player_ids,
        "missing_model_versions": missing_versions,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# Historical ledger reconciliation",
                "",
                f"Generated: {report['generated_at']}",
                "",
                "| Check | Count |",
                "| --- | ---: |",
                f"| Existing JSON ledger rows | {len(rows)} |",
                f"| Dashboard game rows | {len(dashboard_rows)} |",
                f"| Newly imported rows | {imported} |",
                f"| Already-imported idempotent rows | {idempotent} |",
                f"| Canonical database rows | {database_rows} |",
                f"| Exact duplicate source rows | {report['duplicate_rows']} |",
                f"| Games missing from one source | {len(mismatched)} |",
                f"| Outcomes recovered from dashboard | {resolved} |",
                f"| Rows without recoverable outcomes | {unresolvable} |",
                f"| Rows missing player IDs | {missing_player_ids} |",
                f"| Rows missing model versions | {missing_versions} |",
                "",
                "Legacy rows are preserved with explicit quality flags. Only game-winner and",
                "8.5-total outcomes can be reconstructed from the legacy dashboard; NRFI,",
                "starter-strikeout, and batter-HR outcomes require box-score-level migration.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=settings.legacy_ledger_path)
    parser.add_argument("--dashboard", type=Path, default=settings.dashboard_ledger_path)
    parser.add_argument("--report-json", type=Path, default=Path("reports/reconciliation.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/reconciliation.md"))
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.dashboard, args.report_json, args.report_md), indent=2))
