from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from athena_api.database import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("forecast_key", "revision_number", name="uq_prediction_revision"),
        Index("ix_predictions_game_headline", "mlb_game_pk", "is_headline"),
        Index("ix_predictions_target", "category", "statistic", "player_id"),
        Index("ix_predictions_created", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    forecast_key: Mapped[str] = mapped_column(String(255), nullable=False)
    game_id: Mapped[str] = mapped_column(String(128), nullable=False)
    mlb_game_pk: Mapped[int | None] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    statistic: Mapped[str] = mapped_column(String(80), nullable=False)
    player_id: Mapped[int | None] = mapped_column(Integer)
    team_id: Mapped[int | None] = mapped_column(Integer)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    units: Mapped[str | None] = mapped_column(String(32))
    probability: Mapped[float | None] = mapped_column(Float)
    projected_value: Mapped[float | None] = mapped_column(Float)
    interval_low: Mapped[float | None] = mapped_column(Float)
    interval_high: Mapped[float | None] = mapped_column(Float)
    distribution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    distribution_ref: Mapped[str | None] = mapped_column(String(500))
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    git_commit_sha: Mapped[str | None] = mapped_column(String(40))
    data_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)
    data_snapshot_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    rate_source_version: Mapped[str | None] = mapped_column(String(128))
    feature_version: Mapped[str] = mapped_column(String(80), nullable=False)
    simulation_settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    simulation_seed: Mapped[int | None] = mapped_column(Integer)
    simulation_seed_policy: Mapped[str | None] = mapped_column(String(160))
    lineup_player_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    lineup_status: Mapped[str] = mapped_column(String(32), default="unknown")
    starting_pitcher_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    context_feature_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    data_quality_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[str] = mapped_column(String(24), default="low")
    validation_status: Mapped[str] = mapped_column(String(24), default="provisional")
    resolution_status: Mapped[str] = mapped_column(String(24), default="pending")
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    revision_reason: Mapped[str | None] = mapped_column(String(120))
    superseded_prediction_id: Mapped[str | None] = mapped_column(
        ForeignKey("predictions.id")
    )
    validity_status: Mapped[str] = mapped_column(String(24), default="active")
    is_headline: Mapped[bool] = mapped_column(Boolean, default=True)
    first_pitch_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    final_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class DataSnapshot(Base):
    __tablename__ = "data_snapshots"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), default="candidate")
    source_through_date: Mapped[str | None] = mapped_column(String(10))
    schema_version: Mapped[str] = mapped_column(String(32), default="1")
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    checksums: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    validation_results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    promoted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    auth_user_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    display_name: Mapped[str | None] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="America/New_York")
    detail_level: Mapped[str] = mapped_column(String(24), default="balanced")
    default_sort: Mapped[str] = mapped_column(String(32), default="support")
    alert_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
    follows: Mapped[list[Follow]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


class Follow(Base):
    __tablename__ = "follows"
    __table_args__ = (
        UniqueConstraint("profile_id", "entity_type", "entity_id", name="uq_profile_follow"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(160))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    profile: Mapped[UserProfile] = relationship(back_populates="follows")


class AgentAudit(Base):
    __tablename__ = "agent_audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, default=new_uuid)
    auth_user_id: Mapped[str | None] = mapped_column(String(128))
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    provider: Mapped[str] = mapped_column(String(32), default="deterministic")
    model: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(32), default="completed")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
