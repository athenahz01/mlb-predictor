from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from athena_api.models import AgentAudit, Prediction
from athena_api.presentation import prediction_payload
from athena_api.settings import get_settings

_ANSWER_CACHE: dict[tuple, tuple[float, str, str]] = {}


class LanguageProvider(Protocol):
    name: str

    def explain(self, question: str, evidence: dict[str, Any], detail_level: str) -> str: ...


@dataclass
class DeterministicProvider:
    name: str = "deterministic"

    def explain(self, question: str, evidence: dict[str, Any], detail_level: str) -> str:
        prediction = evidence["prediction"]
        value = prediction.get("probability")
        value_text = (
            f"{value:.1%}" if value is not None else f"{prediction['projected_value']:.2f}"
        )
        main = prediction.get("evidence", {}).get("main_reason") or "the current model inputs"
        uncertainty = (
            prediction.get("evidence", {}).get("main_uncertainty")
            or "baseball outcomes remain highly variable"
        )
        warnings = prediction.get("data_quality_flags") or []
        warning_text = (
            f" Data warning: {', '.join(warnings)}."
            if warnings
            else " Data warning: none identified."
        )
        return (
            f"{evidence['label']}: {value_text} ({prediction['confidence']} confidence, "
            f"{prediction['validation_status']}). Main evidence: {main}. "
            f"Main uncertainty: {uncertainty}. Lineup/starter status: "
            f"{prediction['lineup_status']}. Updated {prediction['created_at']}; model "
            f"{prediction['model_version']}.{warning_text}"
        )


def _headline_rows(db: Session, game_id: str | None = None) -> list[Prediction]:
    query = select(Prediction).where(
        Prediction.is_headline.is_(True), Prediction.validity_status == "active"
    )
    if game_id:
        query = query.where(Prediction.game_id == game_id)
    return list(db.scalars(query.order_by(Prediction.created_at.desc())))


def _ranked_prediction(rows: list[Prediction], question: str) -> tuple[Prediction | None, str]:
    q = question.lower()
    selectors = [
        (("homer", "home run"), "home_run_probability", "Home-run outlook"),
        (("strikeout", " k ", "ks"), "starter_strikeouts_over_5_5", "Strikeout outlook"),
        (("hit",), "hit_probability", "Hit outlook"),
        (("total base",), "total_bases", "Total-base outlook"),
        (("runs", "total"), "total_over_8_5", "Run-total outlook"),
        (("win", "winner"), "home_win_probability", "Game winner"),
    ]
    statistic = None
    label = "Prediction"
    for terms, candidate, candidate_label in selectors:
        if any(term in q for term in terms):
            statistic, label = candidate, candidate_label
            break
    if statistic is None and any(term in q for term in ("uncertain", "avoid", "risk")):
        if not rows:
            return None, "Uncertainty outlook"
        return sorted(
            rows,
            key=lambda row: (
                {"low": 0, "medium": 1, "high": 2}.get(row.confidence, 0),
                -len(row.data_quality_flags),
            ),
        )[0], "Uncertainty outlook"
    if statistic is None:
        return None, label
    matches = [
        row
        for row in rows
        if row.statistic == statistic or row.statistic.endswith(statistic)
    ]
    if not matches:
        return None, label
    ranked = sorted(
        matches,
        key=lambda row: (
            row.confidence == "high",
            row.probability if row.probability is not None else row.projected_value or -1,
        ),
        reverse=True,
    )
    return ranked[0], label


def _tool_name(question: str, game_id: str | None) -> str:
    q = question.lower()
    if any(term in q for term in ("change", "move", "moved", "history", "overnight")):
        return "get_prediction_history"
    if "compare" in q and any(term in q for term in ("pitcher", "starter")):
        return "compare_pitchers"
    if "compare" in q:
        return "compare_players"
    if "lineup" in q:
        return "get_lineup_status"
    if any(
        term in q
        for term in ("performance", "performed", "calibration", "model record", "market")
    ):
        return "get_model_performance"
    if game_id:
        return "get_game_prediction"
    return "rank_predictions"


def answer_question(
    db: Session,
    question: str,
    *,
    game_id: str | None = None,
    detail_level: str = "balanced",
    auth_user_id: str | None = None,
    provider: LanguageProvider | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = _headline_rows(db, game_id)
    selected, label = _ranked_prediction(rows, question)
    tool_name = _tool_name(question, game_id)
    if (
        selected is None
        and game_id
        and tool_name in {"get_game_prediction", "get_prediction_history", "get_lineup_status"}
    ):
        selected = next(
            (row for row in rows if row.statistic == "home_win_probability"),
            rows[0] if rows else None,
        )
        label = "Game forecast"
    tool_call = {
        "name": tool_name,
        "arguments": {"game_id": game_id, "question": question},
    }
    request_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    if provider is None:
        from athena_api.providers import provider_from_settings

        provider = provider_from_settings()
    if selected is None:
        answer = (
            "Athena does not have a supported prediction for that request. "
            "I can explain game winners, run totals, starter strikeouts, batter hits, "
            "total bases, and home-run outlooks when those outputs exist."
        )
        grounded = True
    else:
        evidence = {"label": label, "prediction": prediction_payload(selected)}
        cache_key = (
            question.strip().lower(),
            game_id,
            detail_level,
            selected.id,
            provider.name,
        )
        cached = _ANSWER_CACHE.get(cache_key)
        ttl = get_settings().prediction_cache_seconds
        if cached and time.monotonic() - cached[0] <= ttl:
            answer = cached[1]
        else:
            try:
                answer = provider.explain(question, evidence, detail_level)
            except Exception:
                provider = DeterministicProvider()
                answer = provider.explain(question, evidence, detail_level)
            _ANSWER_CACHE[cache_key] = (time.monotonic(), answer, provider.name)
        grounded = True
    audit = AgentAudit(
        auth_user_id=auth_user_id,
        question_hash=request_hash,
        tool_calls=[tool_call],
        provider=provider.name,
        status="completed",
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    db.add(audit)
    db.commit()
    return {
        "answer": answer,
        "tool_calls": [tool_call],
        "grounded": grounded,
        "request_id": audit.request_id,
    }
