"""canonical prediction ledger and user state

Revision ID: 20260723_0001
Revises:
Create Date: 2026-07-23
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260723_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_snapshots",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("source_through_date", sa.String(10)),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("checksums", sa.JSON(), nullable=False),
        sa.Column("validation_results", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("forecast_key", sa.String(255), nullable=False),
        sa.Column("game_id", sa.String(128), nullable=False),
        sa.Column("mlb_game_pk", sa.Integer()),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("statistic", sa.String(80), nullable=False),
        sa.Column("player_id", sa.Integer()),
        sa.Column("team_id", sa.Integer()),
        sa.Column("probability", sa.Float()),
        sa.Column("projected_value", sa.Float()),
        sa.Column("interval_low", sa.Float()),
        sa.Column("interval_high", sa.Float()),
        sa.Column("distribution", sa.JSON()),
        sa.Column("distribution_ref", sa.String(500)),
        sa.Column("model_version", sa.String(80), nullable=False),
        sa.Column("git_commit_sha", sa.String(40)),
        sa.Column("data_snapshot_id", sa.String(128), nullable=False),
        sa.Column("data_snapshot_at", sa.DateTime(timezone=True)),
        sa.Column("rate_source_version", sa.String(128)),
        sa.Column("feature_version", sa.String(80), nullable=False),
        sa.Column("simulation_settings", sa.JSON(), nullable=False),
        sa.Column("simulation_seed", sa.Integer()),
        sa.Column("simulation_seed_policy", sa.String(160)),
        sa.Column("lineup_player_ids", sa.JSON(), nullable=False),
        sa.Column("lineup_status", sa.String(32), nullable=False),
        sa.Column("starting_pitcher_ids", sa.JSON(), nullable=False),
        sa.Column("context_feature_flags", sa.JSON(), nullable=False),
        sa.Column("data_quality_flags", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.String(24), nullable=False),
        sa.Column("validation_status", sa.String(24), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("revision_reason", sa.String(120)),
        sa.Column("superseded_prediction_id", sa.String(36), sa.ForeignKey("predictions.id")),
        sa.Column("validity_status", sa.String(24), nullable=False),
        sa.Column("is_headline", sa.Boolean(), nullable=False),
        sa.Column("first_pitch_at", sa.DateTime(timezone=True)),
        sa.Column("final_result", sa.JSON()),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("forecast_key", "revision_number", name="uq_prediction_revision"),
    )
    op.create_index("ix_predictions_game_headline", "predictions", ["mlb_game_pk", "is_headline"])
    op.create_index(
        "ix_predictions_target", "predictions", ["category", "statistic", "player_id"]
    )
    op.create_index("ix_predictions_created", "predictions", ["created_at"])
    op.create_table(
        "user_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("auth_user_id", sa.String(128), unique=True, nullable=False),
        sa.Column("email", sa.String(320)),
        sa.Column("display_name", sa.String(120)),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("detail_level", sa.String(24), nullable=False),
        sa.Column("default_sort", sa.String(32), nullable=False),
        sa.Column("alert_preferences", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "follows",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "profile_id",
            sa.String(36),
            sa.ForeignKey("user_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(160)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("profile_id", "entity_type", "entity_id", name="uq_profile_follow"),
    )
    op.create_table(
        "agent_audit",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(36), unique=True, nullable=False),
        sa.Column("auth_user_id", sa.String(128)),
        sa.Column("question_hash", sa.String(64), nullable=False),
        sa.Column("tool_calls", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(80)),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("error_code", sa.String(80)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agent_audit")
    op.drop_table("follows")
    op.drop_table("user_profiles")
    op.drop_index("ix_predictions_created", table_name="predictions")
    op.drop_index("ix_predictions_target", table_name="predictions")
    op.drop_index("ix_predictions_game_headline", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("data_snapshots")
