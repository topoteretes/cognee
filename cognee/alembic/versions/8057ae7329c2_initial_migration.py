"""Initial migration

Revision ID: 8057ae7329c2
Revises:
Create Date: 2024-10-02 12:55:20.989372

Historically this revision was empty: the chain is a DELTA log, and the base
tables (users, data, datasets, pipeline_runs, ...) have only ever come from
``Base.metadata.create_all()`` on the startup path. That made the chain
impossible to replay on an empty database — and therefore made "does the chain
agree with the models?" untestable end to end, which is exactly how a chain
shipped ahead of its models went unnoticed until tenants were provisioned with
a schema that disagreed with the revision they were stamped at.

It now creates the base schema from the same metadata the startup path uses,
making the chain a self-sufficient creation history. ``checkfirst=True`` keeps
it a strict no-op for every table that already exists — and every existing
database has this revision recorded as applied, so the body never re-runs
there anyway. It executes only on an ``alembic upgrade head`` against an EMPTY
database: the full-chain lockstep test in CI, and any manual bring-up that
drives alembic directly instead of the startup bootstrap.

Replaying the 42 revisions above this over a fully-built base is safe by
audit, not by hope: every DDL-bearing migration in the chain is
inspector-guarded (verified file by file, and the full replay runs clean —
see ``cognee/tests/e2e/migrations/test_migration_model_lockstep.py``).
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8057ae7329c2"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Imported here, not at module top: alembic imports every versions/ file to
    # build the revision map, and this one must stay importable without pulling
    # the full model tree. By the time the body runs, env.py has already
    # imported cognee, so Base.metadata is populated.
    from cognee.infrastructure.databases.relational import Base

    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    # Deliberately empty: downgrading below the initial revision must not drop
    # user data. ``revert_all_migrations`` guards the destructive paths.
    pass
