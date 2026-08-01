from typing import Set
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import (
    Permission,
    Role,
    RoleDefaultPermissions,
    TenantDefaultPermissions,
    UserRole,
    UserTenant,
)
from cognee.modules.users.permissions.methods.get_tenant import get_tenant
from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES


async def get_effective_capabilities(user_id: UUID, tenant_id: UUID) -> Set[str]:
    """
        Return every capability the user has in the given tenant.

        Capabilities are the union of what the tenant grants all of its members
        and what the user's roles in that tenant grant. There is no deny anywhere
        in this model, so the union is unambiguous and a grant can only widen the
        result.

        Every level is scoped to this tenant and to actual membership in it. That
        matters because the callers of this function guard user management for a
        tenant_id supplied by the request, so an unscoped grant would answer for
        tenants the caller has nothing to do with.

        Capabilities granted directly to a person (UserDefaultPermissions) are
        deliberately not resolved here: that table is keyed on the user alone, so
        honouring it would hand the capability to that person in every tenant in
        the system. Per-person capabilities need a (user, tenant, permission)
        triple that does not exist yet.

        The tenant owner holds every capability in the catalog regardless of what
        is stored, matching how has_user_management_permission already treats them.

        What this resolves is written by the capability endpoints on the
        permissions router, which grant to a tenant or to a role. A tenant with
        no grants yet resolves to an empty set for everyone but the owner, and
        the deprecated role-name fallback in has_user_management_permission is
        what carries such a tenant until its capabilities are assigned.
    Args:
        user_id: Id of the user.
        tenant_id: Id of the tenant the capabilities are scoped to.

    Returns:
        Set[str]: Capability names, empty when the user has none or is not a
        member of the tenant.
    """
    tenant = await get_tenant(tenant_id)

    if tenant.owner_id == user_id:
        return set(CAPABILITY_TYPES)

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        membership = await session.execute(
            select(UserTenant.user_id).where(
                UserTenant.user_id == user_id,
                UserTenant.tenant_id == tenant_id,
            )
        )
        if membership.first() is None:
            return set()

        tenant_level = (
            select(Permission.name)
            .join(
                TenantDefaultPermissions,
                TenantDefaultPermissions.permission_id == Permission.id,
            )
            .where(TenantDefaultPermissions.tenant_id == tenant_id)
        )

        role_level = (
            select(Permission.name)
            .join(
                RoleDefaultPermissions,
                RoleDefaultPermissions.permission_id == Permission.id,
            )
            .join(Role, Role.id == RoleDefaultPermissions.role_id)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user_id, Role.tenant_id == tenant_id)
        )

        capabilities: Set[str] = set()
        for query in (tenant_level, role_level):
            result = await session.execute(query)
            capabilities.update(row[0] for row in result.all())

    # The same table holds dataset permission names (read/write/delete/share);
    # only catalog entries are capabilities.
    return capabilities & CAPABILITY_TYPES
