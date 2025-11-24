from typing import Optional
from uuid import UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.UserTenant import UserTenant
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_tenant
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    TenantNotFoundError,
    PermissionDeniedError,
)


async def add_user_to_tenant(
    user_id: UUID, tenant_id: UUID, owner_id: UUID, set_as_active_tenant: Optional[bool] = False
):
    """
        Add a user with the given id to the tenant with the given id.
        This can only be successful if the request owner with the given id is the tenant owner.

        If set_as_active_tenant is true it will automatically set the users active tenant to provided tenant.
    Args:
        user_id: Id of the user.
        tenant_id: Id of the tenant.
        owner_id: Id of the request owner.
        set_as_active_tenant: If set_as_active_tenant is true it will automatically set the users active tenant to provided tenant.

    Returns:
        None

    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = await get_user(user_id)
        tenant = await get_tenant(tenant_id)

        if not user:
            raise UserNotFoundError
        elif not tenant:
            raise TenantNotFoundError

        if tenant.owner_id != owner_id:
            raise PermissionDeniedError(
                message="Only tenant owner can add other users to organization."
            )

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
