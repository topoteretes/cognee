from typing import Set
from uuid import UUID

from sqlalchemy import or_, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import PrincipalCapability, Role, UserRole, UserTenant
from cognee.modules.users.permissions.methods.get_tenant import get_tenant
from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES


async def get_effective_capabilities(user_id: UUID, tenant_id: UUID) -> Set[str]:
    """
        Return every capability the user has in the given tenant.

        Capabilities are the union of what the tenant grants all of its members,
        what the user's roles in that tenant grant, and what the user was granted
        personally in that tenant. There is no deny anywhere in this model, so
        the union is unambiguous and a grant can only widen the result.

        All three levels are rows in principal_capabilities, keyed on
        (principal, tenant, capability), so the union is one query and a grant
        scoped to another tenant is unrepresentable rather than filtered out.

        Resolution is additionally gated on actual membership in the tenant,
        once, at the top. That matters because the callers of this function
        guard user management for a tenant_id supplied by the request, and it is
        also what makes granting to an invited person safe: the grant may exist
        before they accept, but resolves to nothing until they are a member.

        The tenant owner holds every capability in the catalog regardless of
        what is stored, matching how has_user_management_permission already
        treats them.

        A tenant with no grants yet resolves to an empty set for everyone but
        the owner; the deprecated role-name fallback in
        has_user_management_permission is what carries such a tenant until its
        capabilities are assigned.
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

        user_role_ids = (
            select(UserRole.role_id)
            .join(Role, Role.id == UserRole.role_id)
            .where(UserRole.user_id == user_id, Role.tenant_id == tenant_id)
        )

        result = await session.execute(
            select(PrincipalCapability.capability).where(
                PrincipalCapability.tenant_id == tenant_id,
                or_(
                    PrincipalCapability.principal_id == tenant_id,
                    PrincipalCapability.principal_id == user_id,
                    PrincipalCapability.principal_id.in_(user_role_ids),
                ),
            )
        )
        capabilities: Set[str] = {row[0] for row in result.all()}

    # Grants are validated against the catalog when written, but the catalog is
    # code and can shrink; a name that fell out of it must stop resolving.
    return capabilities & CAPABILITY_TYPES
