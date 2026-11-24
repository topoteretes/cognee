"""Record who revoked an integration connection

Revision ID: e3f1a9c5b7d2
Revises: 91e29f0d0467
Create Date: 2026-11-07 19:00:00.000000

A connection is revoked by its owner's disconnect or by the provider (an app
uninstalled, a token the provider no longer accepts), and both left the same
``status = 'revoked'``, so the Integrations page could not tell its owner that a
connection was lost at the provider. ``revoked_by`` holds ``owner`` or
``provider``; rows revoked before this revision stay NULL, unknown.

Idempotent: the column is added or dropped only when the inspector says so.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3f1a9c5b7d2"
down_revision: str | None = "91e29f0d0467"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "integration_credentials"
COLUMN = "revoked_by"


def _columns() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(TABLE_NAME):
        return set()
    return {column["name"] for column in inspector.get_columns(TABLE_NAME)}


def upgrade() -> None:
    columns = _columns()
    if columns and COLUMN not in columns:
        op.add_column(TABLE_NAME, sa.Column(COLUMN, sa.String(), nullable=True))


def downgrade() -> None:
    if COLUMN in _columns():
        op.drop_column(TABLE_NAME, COLUMN)
