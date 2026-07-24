"""phase 3 parameterized prediction and resolution contract

Revision ID: 20260724_0002
Revises: 20260723_0001
Create Date: 2026-07-24
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260724_0002"
down_revision = "20260723_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("parameters", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column("predictions", sa.Column("units", sa.String(32)))
    op.add_column(
        "predictions",
        sa.Column(
            "resolution_status",
            sa.String(24),
            nullable=False,
            server_default="pending",
        ),
    )


def downgrade() -> None:
    op.drop_column("predictions", "resolution_status")
    op.drop_column("predictions", "units")
    op.drop_column("predictions", "parameters")
