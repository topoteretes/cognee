from uuid import uuid4
from datetime import datetime, timezone
# from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, DateTime, UUID, String
from cognee.infrastructure.databases.relational import Base

class Permission(Base):
    __tablename__ = "permissions"

    id = Column(UUID, primary_key = True, index = True, default = uuid4)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone = True), onupdate = lambda: datetime.now(timezone.utc))

    name = Column(String, unique = True, nullable = False, index = True)

    # acls = relationship("ACL", back_populates = "permission")

    # groups: Mapped[list["Group"]] = relationship(
    #     "Group",
    #     secondary = "group_permissions",
    #     back_populates = "permissions",
    # )
