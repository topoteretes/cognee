from uuid import UUID
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from typing import Optional

from cognee.modules.users.models.UserTenant import UserTenant
from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Tenant
from cognee.modules.users.methods import get_user


async def create_tenant(
    tenant_name: str, user_id: UUID, set_as_active_tenant: Optional[bool] = True
) -> UUID:
    """
        Create a new tenant with the given name, for the user with the given id.
        This user is the owner of the tenant.
    Args:
        tenant_name: Name of the new tenant.
        user_id: Id of the user.
        set_as_active_tenant: If true, set the newly created tenant as the active tenant for the user.

    Returns:
        None
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            user = await get_user(user_id)

            tenant = Tenant(name=tenant_name, owner_id=user_id)
            session.add(tenant)
            await session.flush()

            if set_as_active_tenant:
                user.tenant_id = tenant.id
                await session.merge(user)
                await session.commit()

            try:
                # Add association directly to the association table
                create_user_tenant_statement = insert(UserTenant).values(
                    user_id=user_id, tenant_id=tenant.id
                )
                await session.execute(create_user_tenant_statement)
                await session.commit()
            except IntegrityError:
                raise EntityAlreadyExistsError(message="User is already part of tenant.")

            return tenant.id
        except IntegrityError as e:
            raise EntityAlreadyExistsError(message="Tenant already exists.") from e
