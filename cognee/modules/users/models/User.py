from uuid import UUID as uuid_UUID
from sqlalchemy import ForeignKey, Column, UUID, String
from sqlalchemy.orm import relationship, Mapped
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from .Principal import Principal
from .UserRole import UserRole
from .Role import Role
from fastapi_users import schemas


class User(SQLAlchemyBaseUserTableUUID, Principal):
    __tablename__ = "users"

    id = Column(UUID, ForeignKey("principals.id"), primary_key=True)

    # Many-to-Many Relationship with Roles
    roles: Mapped[list["Role"]] = relationship(
        "Role",
        secondary=UserRole.__tablename__,
        back_populates="users",
    )

    # Foreign key to Tenant (Many-to-One relationship)
    tenant_id = Column(UUID, ForeignKey("tenants.id"), nullable=False)

    # Relationship to Tenant
    tenant = relationship(
        "Tenant",
        back_populates="users",
        foreign_keys=[tenant_id],
    )

    __mapper_args__ = {
        "polymorphic_identity": "user",
    }


# Keep these schemas in sync with User model


class UserRead(schemas.BaseUser[uuid_UUID]):
    pass


class UserCreate(schemas.BaseUserCreate):
    tenant_id: uuid_UUID


class UserUpdate(schemas.BaseUserUpdate):
    pass
