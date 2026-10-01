from sqlalchemy import UUID, Column, ForeignKey, String
from sqlalchemy.orm import Mapped, relationship

from .Principal import Principal
from .Role import Role
from .UserTenant import UserTenant


class Tenant(Principal):
    __tablename__ = "tenants"

    id = Column(UUID, ForeignKey("principals.id"), primary_key=True)
    name = Column(String, unique=False, nullable=False, index=True)

    owner_id = Column(UUID, index=True)

    users: Mapped[list["User"]] = relationship(  # noqa: F821
        "User",
        secondary=UserTenant.__tablename__,
        back_populates="tenants",
    )

    # One-to-Many relationship with Role
    roles = relationship(
        "Role",
        back_populates="tenant",
        foreign_keys=lambda: [Role.tenant_id],
    )

    __mapper_args__ = {
        "polymorphic_identity": "tenant",
    }
