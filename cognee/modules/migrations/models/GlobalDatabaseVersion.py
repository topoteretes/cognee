from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from cognee.infrastructure.databases.relational import Base

# The table has exactly one row, always with this id.
GLOBAL_DATABASE_VERSION_ROW_ID = 1


class GlobalDatabaseVersion(Base):
    """Single-row, deployment-wide version + global-database migration tracking.

    ``cognee_version`` is written on EVERY startup in BOTH access-control modes
    — the one place to read which Cognee release last ran against this
    deployment (the per-dataset ``dataset_database.cognee_version`` records
    only which release last migrated that dataset's databases).

    The ``global_migration_revision`` column applies only when backend access
    control is disabled: in that mode one global graph/vector pair backs every
    dataset (no ``dataset_database`` rows exist to carry revisions), so this
    row tracks the pair's last-applied migration revision. With access control
    enabled it stays NULL — the per-dataset revision lives on ``dataset_database``.

    Standalone on purpose: no Dataset FK and no sentinel rows in user-facing
    tables, so no query anywhere needs to filter anything out.
    """

    __tablename__ = "global_database_version"

    id = Column(Integer, primary_key=True, default=GLOBAL_DATABASE_VERSION_ROW_ID)

    # Cognee release that last started against this deployment (both modes).
    cognee_version = Column(String, nullable=True)

    # Last-applied data-migration revision for the GLOBAL database pair
    # (access control off only; one chain covers graph + vector + ledger).
    # NULL means "no recorded revision" -> all migrations run.
    global_migration_revision = Column(String, nullable=True)
    # Why the last migration attempt failed (NULL when healthy).
    global_migration_last_error = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
