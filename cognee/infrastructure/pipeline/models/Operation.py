from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, MappedColumn
from sqlalchemy import Column, DateTime, ForeignKey, Enum, JSON
from cognee.infrastructure.databases.relational import Base, UUID

class OperationType(Enum):
    MERGE_DATA = "MERGE_DATA"
    APPEND_DATA = "APPEND_DATA"

class OperationStatus(Enum):
    STARTED = "OPERATION_STARTED"
    IN_PROGRESS = "OPERATION_IN_PROGRESS"
    COMPLETE = "OPERATION_COMPLETE"
    ERROR = "OPERATION_ERROR"
    CANCELLED = "OPERATION_CANCELLED"

class Operation(Base):
    __tablename__ = "operation"

    id = Column(UUID, primary_key = True)
    status = Column(Enum(OperationStatus))
    operation_type = Column(Enum(OperationType))

    data_id = Column(UUID, ForeignKey("data.id"))
    meta_data: Mapped[dict] = MappedColumn(type_ = JSON)

    created_at = Column(DateTime, default = datetime.now(timezone.utc))
