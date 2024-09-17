from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, ForeignKey, DateTime
from cognee.infrastructure.databases.relational import Base, UUID
from .ACLResources import ACLResources

class ACL(Base):
    __tablename__ = "acls"

    id = Column(UUID, primary_key = True, default = uuid4)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone = True), onupdate = lambda: datetime.now(timezone.utc))

    principal_id = Column(UUID, ForeignKey("principals.id"))
    permission_id = Column(UUID, ForeignKey("permissions.id"))

    principal = relationship("Principal")
    permission = relationship("Permission")
    resources: Mapped[list["Resource"]] = relationship(
        "Resource",
        secondary = ACLResources.__tablename__,
        back_populates = "acls",
    )
