"""Initial migration

Revision ID: 8057ae7329c2
Revises:
Create Date: 2024-10-02 12:55:20.989372

Historically this revision was EMPTY: the chain was a delta log and the base
tables only ever came from ``Base.metadata.create_all()`` + ``stamp head`` —
an unverified assertion that the models and the chain describe the same
schema (the assertion whose silent break caused the Aug-2026 cloud mis-stamp
incident). It now carries the FROZEN base schema, making the chain a
self-sufficient creation history: ``alembic upgrade head`` on an empty
database builds the real schema, so the fresh-database bootstrap runs the
chain instead of stamping, and ``alembic_version`` only ever records
revisions that actually executed.

The schema comes from ``cognee.alembic.frozen_schema`` — a frozen snapshot
rendered by Alembic's autogenerate engine from the certified model surface at
chain head d1e2f3a4b5c6 (2026-09-01; a real chain-migrated database was
verified to match the models exactly), shared with the reconcile revision
1c22e6cb5aec so the two can never diverge. FROZEN means frozen: never
regenerate it from live models — a future model change ships as a NEW
migration at the head of the chain, exactly as before. Every table (with its
indexes) is created only when absent, so this revision is idempotent, safe on
partially-existing legacy databases, and inert on any database where it is
already recorded (it never runs there — Alembic considers it applied).
"""

from typing import Sequence, Union

from alembic import op

from cognee.alembic.frozen_schema import (
    create_enum_types_if_missing,
    create_missing_tables,
    frozen_metadata,
)

# revision identifiers, used by Alembic.
revision: str = "8057ae7329c2"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create every frozen table that does not exist yet — the base schema on
    an empty database, a no-op per table on one that already has it. The
    surface itself lives in ``cognee.alembic.frozen_schema`` (FROZEN: see its
    docstring); this revision only decides what is missing."""
    bind = op.get_bind()
    create_enum_types_if_missing(bind)
    create_missing_tables(bind, frozen_metadata(bind.dialect.name))


def downgrade() -> None:
    # Downgrade-to-base never drops the base schema (the historical contract).
    pass
