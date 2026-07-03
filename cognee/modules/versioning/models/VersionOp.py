from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import JSON, UUID, Column, DateTime, String

from cognee.infrastructure.databases.relational import Base


class VersionOpStatus:
    """Lifecycle of a destructive versioning operation.

    The inverse payload is committed (``CAPTURED``) *before* the destructive
    graph/vector mutation runs. Graph, vector, and relational stores cannot
    share one ACID transaction, so this write-ahead order plus an idempotent
    restore is the recovery guarantee: a crash mid-operation leaves a
    ``CAPTURED`` row whose inverse can be replayed safely.
    """

    CAPTURED = "captured"
    APPLIED = "applied"
    UNDONE = "undone"


class VersionOp(Base):
    """Write-ahead undo ledger for destructive versioning operations.

    Each row records one reversible operation (a ``forget`` or a ``rollback``)
    with the exact inverse needed to restore the destroyed graph artifacts,
    their provenance columns, and their vector rows. Rows past ``expires_at``
    (the retention window) may be pruned, after which the operation is
    permanently irreversible.
    """

    __tablename__ = "version_ops"

    id = Column(UUID, primary_key=True, default=uuid4)

    dataset_id = Column(UUID, index=True, nullable=False)
    op_type = Column(String, nullable=False)  # "forget" | "rollback"
    status = Column(String, nullable=False, default=VersionOpStatus.CAPTURED)

    # Versioned inverse payload; see modules/versioning/methods/inverse.py.
    payload = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=True)
