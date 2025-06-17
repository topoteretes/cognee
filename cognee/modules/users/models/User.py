from typing import Optional
from uuid import UUID as uuid_UUID
from fastapi_users import schemas
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import ForeignKey, Column, UUID
from sqlalchemy.orm import relationship, Mapped

from .Principal import Principal
from .UserRole import UserRole
from .Role import Role


class User(SQLAlchemyBaseUserTableUUID, Principal):
    __tablename__ = "users"

    id = Column(UUID, ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True)

    # Foreign key to Tenant (Many-to-One relationship)
    tenant_id = Column(UUID, ForeignKey("tenants.id"))

    # Many-to-Many Relationship with Roles
    roles: Mapped[list["Role"]] = relationship(
        "Role",
        secondary=UserRole.__tablename__,
        back_populates="users",
    )

    # Relationship to Tenant
    tenant = relationship(
        "Tenant",
        back_populates="users",
        foreign_keys=[tenant_id],
    )

    # ACL Relationship (One-to-Many)
    acls = relationship("ACL", back_populates="principal", cascade="all, delete")

    __mapper_args__ = {
        "polymorphic_identity": "user",
    }


# Keep these schemas in sync with User model
class UserRead(schemas.BaseUser[uuid_UUID]):
    tenant_id: Optional[uuid_UUID] = None


class UserCreate(schemas.BaseUserCreate):
    tenant_id: Optional[uuid_UUID] = None
    is_verified: bool = True


class UserUpdate(schemas.BaseUserUpdate):
    pass
