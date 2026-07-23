from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionCreate(BaseModel):
    game_id: str
    mlb_game_pk: int | None = None
    category: Literal["game", "team", "pitcher", "batter"]
    statistic: str
    player_id: int | None = None
    team_id: int | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    projected_value: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    distribution: dict[str, Any] | None = None
    distribution_ref: str | None = None
    model_version: str
    git_commit_sha: str | None = None
    data_snapshot_id: str
    data_snapshot_at: dt.datetime | None = None
    rate_source_version: str | None = None
    feature_version: str
    simulation_settings: dict[str, Any] = Field(default_factory=dict)
    simulation_seed: int | None = None
    simulation_seed_policy: str | None = None
    lineup_player_ids: dict[str, Any] = Field(default_factory=dict)
    lineup_status: str = "unknown"
    starting_pitcher_ids: dict[str, Any] = Field(default_factory=dict)
    context_feature_flags: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: Literal["low", "medium", "high"] = "low"
    validation_status: Literal["validated", "provisional", "experimental", "unavailable"] = (
        "provisional"
    )
    revision_reason: str | None = None
    first_pitch_at: dt.datetime | None = None
    created_at: dt.datetime | None = None

    @model_validator(mode="after")
    def validate_value(self) -> PredictionCreate:
        if self.probability is None and self.projected_value is None:
            raise ValueError("probability or projected_value is required")
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("prediction intervals require both bounds")
        return self


class PredictionRead(PredictionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    forecast_key: str
    revision_number: int
    superseded_prediction_id: str | None
    validity_status: str
    is_headline: bool
    final_result: dict[str, Any] | None
    resolved_at: dt.datetime | None
    created_at: dt.datetime


class PredictionResolve(BaseModel):
    result: dict[str, Any]
    resolved_at: dt.datetime | None = None


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    timezone: str | None = Field(default=None, max_length=64)
    detail_level: Literal["beginner", "balanced", "advanced"] | None = None
    default_sort: Literal["support", "time", "confidence"] | None = None
    alert_preferences: dict[str, Any] | None = None


class FollowCreate(BaseModel):
    entity_type: Literal["team", "batter", "pitcher", "prediction_category"]
    entity_id: str
    display_name: str | None = None


class AgentQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    game_id: str | None = None
    detail_level: Literal["beginner", "balanced", "advanced"] = "balanced"


class AgentAnswer(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]]
    grounded: bool
    request_id: str
