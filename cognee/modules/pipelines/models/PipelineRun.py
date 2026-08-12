import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, JSON, Enum, UUID, String
from cognee.infrastructure.databases.relational import Base


class PipelineRunStatus(enum.Enum):
    DATASET_PROCESSING_INITIATED = "DATASET_PROCESSING_INITIATED"
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    status = Column(Enum(PipelineRunStatus))
    pipeline_run_id = Column(UUID, index=True)
    pipeline_name = Column(String)
    pipeline_id = Column(UUID, index=True)
    dataset_id = Column(UUID, index=True)
    run_info = Column(JSON)

    # Wall-clock stamp of the last time this run finished a pipeline task.
    # This table is an append-only event log -- every status change INSERTs a
    # new row sharing one pipeline_run_id -- so created_at records transitions
    # but nothing moves while a run is working. This is the one field that
    # advances mid-run, and it only carries meaning on the
    # DATASET_PROCESSING_STARTED row of a run.
    #
    # NULL means "no progress recorded yet": a run that has not completed its
    # first task, or a row written before this column existed. Readers fall
    # back to created_at in that case.
    last_heartbeat_at = Column(DateTime(timezone=True), nullable=True)

    # Which process owns this run, so liveness can be a fact rather than an
    # inference. A heartbeat only ever says "it was alive N seconds ago"; the
    # owner says "ask the operating system". Recorded on the STARTED row.
    #
    # owner_node_id defaults to the hostname, which is deliberate: processes
    # that share a hostname share a process table, so owner_pid is meaningful
    # exactly when the ids match. Containers get distinct hostnames and
    # correctly fall through to the heartbeat instead of reading each other's
    # pids. Override with COGNEE_NODE_ID when hostname is not stable per host.
    owner_node_id = Column(String, nullable=True)
    owner_pid = Column(Integer, nullable=True)
