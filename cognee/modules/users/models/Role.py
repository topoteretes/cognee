from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, String, ForeignKey, UUID, UniqueConstraint
from .Principal import Principal
from .UserRole import UserRole


class Role(Principal):
    __tablename__ = "roles"

    id = Column(UUID, ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True)

    name = Column(String, nullable=False, index=True)

    users: Mapped[list["User"]] = relationship(  # noqa: F821
        "User",
        secondary=UserRole.__tablename__,
        back_populates="roles",
    )

    # Foreign key to Tenant (Many-to-One relationship)
    tenant_id = Column(UUID, ForeignKey("tenants.id"), nullable=False)

    # Relationship to Tenant
    tenant = relationship("Tenant", back_populates="roles", foreign_keys=[tenant_id])

    # Unique constraint on tenant_id and name
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_id_name"),)

    __mapper_args__ = {
        "polymorphic_identity": "role",
    }
