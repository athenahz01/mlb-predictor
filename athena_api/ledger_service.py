from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from athena_api.models import Prediction, utcnow
from athena_api.schemas import PredictionCreate


def forecast_key(data: PredictionCreate) -> str:
    parts = [
        data.game_id,
        data.category,
        data.statistic,
        str(data.player_id or ""),
        str(data.team_id or ""),
    ]
    return "|".join(parts)


def _fingerprint(data: PredictionCreate) -> str:
    payload = data.model_dump(mode="json", exclude={"revision_reason", "created_at"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _latest_query(key: str) -> Select[tuple[Prediction]]:
    return (
        select(Prediction)
        .where(Prediction.forecast_key == key)
        .order_by(Prediction.revision_number.desc())
        .limit(1)
    )


def create_revision(db: Session, data: PredictionCreate) -> tuple[Prediction, bool]:
    """Create an immutable revision, returning (prediction, created).

    Repeating the exact same payload is idempotent. A material update marks the
    previous headline as superseded but never edits its forecast values.
    """
    key = forecast_key(data)
    previous = db.scalar(_latest_query(key))
    fingerprint = _fingerprint(data)
    if previous and previous.evidence.get("_input_fingerprint") == fingerprint:
        return previous, False

    evidence = dict(data.evidence)
    evidence["_input_fingerprint"] = fingerprint
    if previous:
        previous.is_headline = False
        if previous.validity_status == "active":
            previous.validity_status = "superseded"

    values = data.model_dump(exclude={"created_at"})
    values["forecast_key"] = key
    values["evidence"] = evidence
    values["revision_number"] = (previous.revision_number + 1) if previous else 1
    values["superseded_prediction_id"] = previous.id if previous else None
    values["is_headline"] = True
    values["validity_status"] = "active"
    if data.created_at is not None:
        values["created_at"] = data.created_at
    prediction = Prediction(**values)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction, True


def invalidate_prediction(db: Session, prediction: Prediction, reason: str) -> Prediction:
    prediction.validity_status = "invalid"
    prediction.is_headline = False
    prediction.data_quality_flags = sorted(
        set([*prediction.data_quality_flags, f"invalid:{reason}"])
    )
    db.commit()
    db.refresh(prediction)
    return prediction


def resolve_prediction(
    db: Session, prediction: Prediction, result: dict[str, Any], resolved_at=None
) -> Prediction:
    if prediction.final_result is not None:
        if prediction.final_result != result:
            raise ValueError("prediction already resolved with a different result")
        return prediction
    prediction.final_result = result
    prediction.resolved_at = resolved_at or utcnow()
    db.commit()
    db.refresh(prediction)
    return prediction


def prediction_tracks(
    db: Session, *, game_id: str, statistic: str | None = None
) -> dict[str, list[Prediction]]:
    filters = [Prediction.game_id == game_id]
    if statistic:
        filters.append(Prediction.statistic == statistic)
    rows = list(
        db.scalars(
            select(Prediction)
            .where(and_(*filters))
            .order_by(Prediction.forecast_key, Prediction.revision_number)
        )
    )
    initial: dict[str, Prediction] = {}
    latest: dict[str, Prediction] = {}
    for row in rows:
        initial.setdefault(row.forecast_key, row)
        if row.validity_status != "invalid":
            latest[row.forecast_key] = row
    return {"initial": list(initial.values()), "latest_pregame": list(latest.values())}
