"""SQLAlchemy model for a single memory-accuracy-score run."""

import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UUID,
)

from cognee.infrastructure.databases.relational import Base


class MemoryScoreRunStatus(enum.Enum):
    INITIATED = "INITIATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERRORED = "ERRORED"
    SKIPPED_INSUFFICIENT_DATA = "SKIPPED_INSUFFICIENT_DATA"


class MemoryScoreRun(Base):
    """One memory-accuracy-score evaluation of a tenant's graph.

    Scoped to ONE dataset, named by the caller. Under
    ``ENABLE_BACKEND_ACCESS_CONTROL`` each user+dataset pair has its own graph
    database, so "the tenant's graph" is not a single graph and the dataset to
    score cannot be inferred — the caller states it (``default_dataset`` on
    Cloud, ``main_dataset`` in OSS). ``tenant_id`` scopes ownership of the run
    row itself.

    ``triggered_by_user_id`` is nullable because a run can be started by a
    scheduler with no acting user. When set it is also whose real question
    history was replayed — query text is one member's search history and is
    never read tenant-wide.

    ``overall_accuracy`` is computed from SYNTHETIC questions only —
    real questions have no golden answer and only carry a groundedness
    boolean, so the two signals are never averaged together.

    ``below_data_floor`` / ``floor_reason`` / ``schema_defined`` are raw
    signals. No call-to-action is derived here; thresholds and copy are
    the UI's job.

    ``topics`` holds the per-topic aggregate as JSON:
    ``[{"topic", "accuracy", "synthetic_count", "real_count",
    "from_real_traffic"}, ...]``.
    """

    __tablename__ = "memory_score_runs"

    __table_args__ = (
        Index("ix_memory_score_runs_tenant_id_created_at", "tenant_id", "created_at"),
    )

    id = Column(UUID, primary_key=True, default=uuid4)

    tenant_id = Column(UUID, index=True)

    # The dataset actually scored. Required input, never inferred — see the
    # class docstring. Recorded so the UI can always name what was measured.
    dataset_id = Column(UUID, index=True)

    triggered_by_user_id = Column(UUID, nullable=True)

    status = Column(Enum(MemoryScoreRunStatus))

    # Data-floor gate, evaluated before any LLM tokens are spent.
    below_data_floor = Column(Boolean, default=False)
    floor_reason = Column(String, nullable=True)

    schema_defined = Column(Boolean, default=False)

    # Synthetic-only aggregate.
    overall_accuracy = Column(Float, nullable=True)

    synthetic_question_count = Column(Integer, default=0)
    real_question_count = Column(Integer, default=0)

    topics = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
