from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, MappedColumn
from sqlalchemy import Column, DateTime, ForeignKey, Enum, JSON
from cognee.infrastructure.databases.relational import Base, UUID


class OperationType(Enum):
    """
    Define various types of operations for data handling.

    Public methods:
    - __str__(): Returns a string representation of the operation type.

    Instance variables:
    - MERGE_DATA: Represents the merge data operation type.
    - APPEND_DATA: Represents the append data operation type.
    """

    MERGE_DATA = "MERGE_DATA"
    APPEND_DATA = "APPEND_DATA"


class OperationStatus(Enum):
    """
    Represent the status of an operation with predefined states.
    """

    STARTED = "OPERATION_STARTED"
    IN_PROGRESS = "OPERATION_IN_PROGRESS"
    COMPLETE = "OPERATION_COMPLETE"
    ERROR = "OPERATION_ERROR"
    CANCELLED = "OPERATION_CANCELLED"


class Operation(Base):
    """
    Represents an operation in the system, extending the Base class.

    This class defines the structure of the 'operation' table, including fields for the
    operation's ID, status, type, associated data, metadata, and creation timestamp. The
    public methods available in this class are inherited from the Base class. Instance
    variables include:
    - id: Unique identifier for the operation.
    - status: The current status of the operation.
    - operation_type: The type of operation being represented.
    - data_id: Foreign key referencing the associated data's ID.
    - meta_data: Additional metadata related to the operation.
    - created_at: Timestamp for when the operation was created.
    """

    __tablename__ = "operation"

    id = Column(UUID, primary_key=True)
    status = Column(Enum(OperationStatus))
    operation_type = Column(Enum(OperationType))

    data_id = Column(UUID, ForeignKey("data.id"))
    meta_data: Mapped[dict] = MappedColumn(type_=JSON)

    created_at = Column(DateTime, default=datetime.now(timezone.utc))
