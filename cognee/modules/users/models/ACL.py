from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class ACL(Base):
    __tablename__ = "acls"

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    principal_id = Column(UUID, ForeignKey("principals.id"))
    permission_id = Column(UUID, ForeignKey("permissions.id"))
    data_id = Column(UUID, ForeignKey("data.id", ondelete="CASCADE"))

    principal = relationship("Principal")
    permission = relationship("Permission")
    data = relationship("Data", back_populates="acls")
