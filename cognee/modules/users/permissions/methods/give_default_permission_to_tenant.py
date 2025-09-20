from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    TenantNotFoundError,
)
from cognee.modules.users.models import (
    Permission,
    Tenant,
    TenantDefaultPermissions,
)


async def give_default_permission_to_tenant(tenant_id: UUID, permission_name: str):
    """
        Give the permission with given name to the tenant with the given id as a default permission.
    Args:
        tenant_id: Id of the tenant
        permission_name: Name of the permission

    Returns:
        None
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        tenant = (
            (await session.execute(select(Tenant).where(Tenant.id == tenant_id))).scalars().first()
        )

        if not tenant:
            raise TenantNotFoundError

        permission_entity = (
            (await session.execute(select(Permission).where(Permission.name == permission_name)))
            .scalars()
            .first()
        )

        if not permission_entity:
            create_permission_statement = insert(Permission).values(name=permission_name)
            await session.execute(create_permission_statement)
            permission_entity = (
                (
                    await session.execute(
                        select(Permission).where(Permission.name == permission_name)
                    )
                )
                .scalars()
                .first()
            )

        try:
            # add default permission to tenant
            await session.execute(
                insert(TenantDefaultPermissions).values(
                    tenant_id=tenant.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Tenant permission already exists.")

        await session.commit()
