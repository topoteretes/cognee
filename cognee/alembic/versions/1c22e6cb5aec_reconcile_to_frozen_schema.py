"""Reconcile every database to the chain's frozen truth

Revision ID: 1c22e6cb5aec
Revises: f3a7b9c1d2e4
Create Date: 2026-09-02 12:34:49.529355

A database bootstrapped by the pre-chain-born code (``create_all`` + ``stamp
head``, every release up to 1.5.3) contains exactly the tables the CREATING
process had imported — ``import cognee`` registers 30 of the 32 model tables;
``integration_credentials`` and ``sync_operations`` are only registered by the
API routers — and the head stamp means the chain's own creation migrations
(b2c4d6e8f0a1, 211ab850ef3d) never run for it. Nothing heals such a database
afterwards: the initial revision counts as applied, and ``create_database()``
now runs the chain (which no-ops at head) instead of an incremental
``create_all``.

This revision converges every database to the chain's frozen truth,
additively and idempotently: every table missing is created whole (with its
indexes and constraints), every column and index missing from an existing
table is added, nothing is dropped or altered. Unique and foreign-key
CONSTRAINTS missing from an EXISTING table are out of scope (a foreign key
rides along only with the column it belongs to). Chain-born databases contain
all of it already and pass through as a no-op. The surface is
``cognee.alembic.frozen_schema`` — the same frozen snapshot the initial
revision (8057ae7329c2) builds from, so the two can never diverge: the
certified model surface at head d1e2f3a4b5c6 plus the one chain-provided
index that surface lacks (``ix_pipeline_runs_dataset_pipeline_created_at``
from d1e2f3a4b5c6: ``PipelineRun.__table_args__`` is assigned twice, so the
model never declared it). FROZEN means frozen: never regenerate from live
models; future schema changes ship as new revisions at the head.

A column that is NOT NULL without a server default cannot be added to a
populated table without a backfill this revision cannot know; that case is
checked up front and aborts the migration BEFORE any change, naming the
columns, so it gets a dedicated backfill migration instead of a silent
nullable compromise. On SQLite, ``ADD COLUMN`` rejects non-constant defaults
(``now()``), so column adds go through Alembic's batch mode, rebuilding the
table only when such a default is involved.
"""

from typing import Sequence, Union

from alembic import op

from cognee.alembic.frozen_schema import frozen_metadata, reconcile

# revision identifiers, used by Alembic.
revision: str = "1c22e6cb5aec"
down_revision: Union[str, None] = "f3a7b9c1d2e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Converge this database to the frozen surface in
    ``cognee.alembic.frozen_schema``: missing tables created whole, missing
    columns and indexes added, nothing dropped or altered; plans first and
    aborts before any change on a column it cannot add safely."""
    reconcile(op, frozen_metadata(op.get_bind().dialect.name))


def downgrade() -> None:
    # Reconciliation only ever adds what the chain already promises; there is
    # nothing to revert that the rest of the chain would not also own.
    pass
