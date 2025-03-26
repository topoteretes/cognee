from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import UUID, Column, DateTime, String, Index
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base


class Relationship(Base):
    __tablename__ = "relationships"

    id = Column(UUID, primary_key=True, default=uuid4)
    parent_id = Column(UUID, nullable=False)
    child_id = Column(UUID, nullable=False)
    creator_function = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    user_id = Column(UUID, nullable=False)

    # Create indexes
    __table_args__ = (
        Index("idx_relationships_parent_id", "parent_id"),
        Index("idx_relationships_child_id", "child_id"),
    )

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "parent_id": str(self.parent_id),
            "child_id": str(self.child_id),
            "creator_function": self.creator_function,
            "created_at": self.created_at.isoformat(),
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "user_id": str(self.user_id),
        }
