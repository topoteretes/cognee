from uuid import UUID
from typing import Union

import sqlalchemy.exc
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.UserTenant import UserTenant
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_tenant
from cognee.modules.users.exceptions import UserNotFoundError, TenantNotFoundError


async def select_tenant(user_id: UUID, tenant_id: Union[UUID, None]):
    """
        Set the users active tenant to provided tenant.

        If None tenant_id is provided set current Tenant to the default single user-tenant
    Args:
        user_id: Id of the user.
        tenant_id: Id of the tenant.

    Returns:
        None

    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = await get_user(user_id)

        if tenant_id is None:
            # If no tenant_id is provided set current Tenant to the single user-tenant
            user.tenant_id = None
            await session.merge(user)
            await session.commit()
            return

        tenant = await get_tenant(tenant_id)

        if not user:
            raise UserNotFoundError
        elif not tenant:
            raise TenantNotFoundError

        # Check if User is part of Tenant
        result = await session.execute(
            select(UserTenant)
            .where(UserTenant.user_id == user_id)
            .where(UserTenant.tenant_id == tenant_id)
        )

        try:
            result = result.scalar_one()
        except sqlalchemy.exc.NoResultFound as e:
            raise TenantNotFoundError("User Tenant relationship not found.") from e

        if result:
            # If user is part of tenant update current tenant of user
            user.tenant_id = tenant_id
            await session.merge(user)
            await session.commit()
