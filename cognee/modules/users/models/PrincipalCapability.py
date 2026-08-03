from datetime import datetime, timezone

from sqlalchemy import UUID, Column, DateTime, ForeignKey, String

from cognee.infrastructure.databases.relational import Base


class PrincipalCapability(Base):
    """One capability held by one principal inside one tenant.

    Capabilities are tenant-scoped actions ("manage_users"), as opposed to the
    dataset permissions the ACL carries. They get their own table rather than
    the ``*DefaultPermissions`` ones for two reasons: those tables reference the
    dataset-permission catalog, so capability names would live next to
    read/write/delete/share and have to be told apart by string filtering, and
    the user-level one carries no tenant column, so a per-person grant would
    apply in every tenant the person belongs to.

    ``tenant_id`` is stored on every row, including those whose principal is a
    role or the tenant itself, where it is derivable. The redundancy is the
    point: resolution filters on it directly, and a grant leaking across
    tenants becomes unrepresentable instead of merely forbidden.
    """

    __tablename__ = "principal_capabilities"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    principal_id = Column(UUID, ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True)

    tenant_id = Column(UUID, ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)

    # A name from the CAPABILITY_TYPES catalog, not a foreign key: the catalog
    # is code, because the code is what gives each name meaning.
    capability = Column(String, primary_key=True)
