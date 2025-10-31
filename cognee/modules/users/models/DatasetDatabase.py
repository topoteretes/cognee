from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, UUID, ForeignKey
from cognee.infrastructure.databases.relational import Base


class DatasetDatabase(Base):
    __tablename__ = "dataset_database"

    owner_id = Column(UUID, ForeignKey("principals.id", ondelete="CASCADE"), index=True)
    dataset_id = Column(
        UUID, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True, index=True
    )

    vector_database_name = Column(String, unique=True, nullable=False)
    graph_database_name = Column(String, unique=True, nullable=False)

    vector_database_provider = Column(String, unique=False, nullable=False)
    graph_database_provider = Column(String, unique=False, nullable=False)

    vector_database_url = Column(String, unique=False, nullable=True)
    graph_database_url = Column(String, unique=False, nullable=True)

    vector_database_key = Column(String, unique=False, nullable=True)
    graph_database_key = Column(String, unique=False, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
