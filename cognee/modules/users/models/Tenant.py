from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, String, ForeignKey, UUID
from .Principal import Principal
from .UserTenant import UserTenant
from .Role import Role


class Tenant(Principal):
    __tablename__ = "tenants"

    id = Column(UUID, ForeignKey("principals.id"), primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)

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
