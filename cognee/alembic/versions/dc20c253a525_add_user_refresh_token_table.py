"""Add the user_refresh_token table

Revision ID: dc20c253a525
Revises: 3cb1c5e71f26
Create Date: 2026-11-27 06:57:53.251076

A browser login ended with its access token (one hour by default). Login now
also issues a refresh token that the web UI exchanges for a new access token;
this table holds each token's hash, the fingerprint of the password it was
issued under and the end of its allowed idle time.

Idempotent: the table is created or dropped only when the inspector says so.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc20c253a525"
down_revision: str | None = "3cb1c5e71f26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "user_refresh_token"


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE_NAME):
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.UUID, primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID,
            sa.ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("token_hash", sa.String, nullable=False, unique=True, index=True),
        sa.Column("password_fingerprint", sa.String, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table(TABLE_NAME):
        op.drop_table(TABLE_NAME)
