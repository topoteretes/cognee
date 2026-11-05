from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import PermissionDeniedError, TenantNotFoundError
from cognee.modules.users.models import UserTenant
from cognee.modules.users.permissions.methods.get_tenant import get_tenant


async def require_tenant_membership(user_id: UUID, tenant_id: UUID) -> bool:
    """
        Assert that the user belongs to the tenant, for endpoints that take a
        tenant_id straight from the request.

        A missing tenant raises the same error as a tenant the caller has nothing
        to do with. Distinguishing them would let any authenticated user probe
        which tenant ids exist by comparing 404 against a successful response.

        This is deliberately separate from get_effective_capabilities, which
        answers a query and returns an empty set rather than raising. Both check
        membership because one is the authorization boundary and the other must
        never widen it.
    Args:
        user_id: Id of the user making the request.
        tenant_id: Id of the tenant taken from the request.

    Returns:
        True when the user owns or belongs to the tenant.

    Raises:
        PermissionDeniedError: The user is not a member, or the tenant does not
            exist.
    """
    denied = PermissionDeniedError(message="User is not a member of this tenant")

    try:
        tenant = await get_tenant(tenant_id)
    except TenantNotFoundError:
        raise denied

    if tenant.owner_id == user_id:
        return True

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        membership = await session.execute(
            select(UserTenant.user_id).where(
                UserTenant.user_id == user_id,
                UserTenant.tenant_id == tenant_id,
            )
        )
        if membership.first() is None:
            raise denied

    return True
