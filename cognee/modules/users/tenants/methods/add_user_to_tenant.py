from uuid import UUID

from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    TenantNotFoundError,
    UserNotFoundError,
)
from cognee.modules.users.methods import get_user
from cognee.modules.users.models.UserTenant import UserTenant
from cognee.modules.users.permissions.methods import get_tenant, has_user_management_permission


async def add_user_to_tenant(
    user_id: UUID, tenant_id: UUID, owner_id: UUID, set_as_active_tenant: bool | None = False
):
    """
        Add a user with the given id to the tenant with the given id.
        This can only be successful if the request owner with the given id can
        manage users in the tenant. The tenant owner always can.

        If set_as_active_tenant is true it will automatically set the users active tenant to provided tenant.
    Args:
        user_id: Id of the user.
        tenant_id: Id of the tenant.
        owner_id: Id of the request owner.
        set_as_active_tenant: If set_as_active_tenant is true it will automatically set the users active tenant to provided tenant.

    Returns:
        None

    Raises:
        UserNotFoundError: If the user does not exist.
        TenantNotFoundError: If the tenant does not exist.
        PermissionDeniedError: If the request owner cannot manage users in the
            tenant.
    """
    db_engine = get_relational_engine()

    # Resolve user + tenant and check permission (each opens its own session)
    # BEFORE opening ours, so this request never holds two pooled connections
    # at once: that overlap deadlocks the pool under concurrency (issue #4197
    # class).
    user = await get_user(user_id)
    tenant = await get_tenant(tenant_id)

    if not user:
        raise UserNotFoundError
    elif not tenant:
        raise TenantNotFoundError

    await has_user_management_permission(requester_id=owner_id, tenant_id=tenant_id)

    async with db_engine.get_async_session() as session:
        if set_as_active_tenant:
            user.tenant_id = tenant_id
            await session.merge(user)
            await session.commit()

        try:
            # Add association directly to the association table
            create_user_tenant_statement = insert(UserTenant).values(
                user_id=user_id, tenant_id=tenant_id
            )
            await session.execute(create_user_tenant_statement)
            await session.commit()

        except IntegrityError:
            raise EntityAlreadyExistsError(message="User is already part of group.")
