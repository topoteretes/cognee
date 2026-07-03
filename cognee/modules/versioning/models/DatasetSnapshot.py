from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import UUID, Column, DateTime, String, UniqueConstraint

from cognee.infrastructure.databases.relational import Base


class DatasetSnapshot(Base):
    """A named cursor into a dataset's pipeline-run ledger.

    A snapshot copies no data: it records the wall-clock cut (``as_of_time``)
    and, for convenience, the newest completed ``pipeline_run_id`` at that cut.
    Time-travel reads resolve a snapshot name to its ``as_of_time`` and filter
    artifacts by the runs completed up to that moment.
    """

    __tablename__ = "dataset_snapshots"
    __table_args__ = (UniqueConstraint("dataset_id", "name", name="uq_dataset_snapshot_name"),)

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(String, nullable=False)
    dataset_id = Column(UUID, index=True, nullable=False)

    as_of_time = Column(DateTime(timezone=True), nullable=False)
    latest_pipeline_run_id = Column(UUID, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
