from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, DateTime, Uuid
from cognee.infrastructure.databases.relational import Base


class ACL(Base):
    __tablename__ = "acls"

    id = Column(Uuid, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    principal_id = Column(Uuid, ForeignKey("principals.id"))
    permission_id = Column(Uuid, ForeignKey("permissions.id"))
    dataset_id = Column(Uuid, ForeignKey("datasets.id", ondelete="CASCADE"))

    principal = relationship("Principal")
    permission = relationship("Permission")
    dataset = relationship("Dataset", back_populates="acls")
