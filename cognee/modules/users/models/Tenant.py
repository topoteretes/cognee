from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, ForeignKey, UUID
from .Principal import Principal
from .User import User
from .Role import Role


class Tenant(Principal):
    __tablename__ = "tenants"

    id = Column(UUID, ForeignKey("principals.id"), primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)

    owner_id = Column(UUID, index=True)

    # One-to-Many relationship with User; specify the join via User.tenant_id
    users = relationship(
        "User",
        back_populates="tenant",
        foreign_keys=lambda: [User.tenant_id],
    )

    # One-to-Many relationship with Role (if needed; similar fix)
    roles = relationship(
        "Role",
        back_populates="tenant",
        foreign_keys=lambda: [Role.tenant_id],
    )

    __mapper_args__ = {
        "polymorphic_identity": "tenant",
    }
