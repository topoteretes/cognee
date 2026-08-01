from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    PermissionNotFoundError,
    TenantNotFoundError,
)
from cognee.modules.users.models import (
    Permission,
    Tenant,
    TenantDefaultPermissions,
)


async def revoke_default_permission_from_tenant(tenant_id: UUID, permission_name: str):
    """
        Take the named default permission away from the tenant with the given id.

        Revoking is idempotent: a permission the tenant never had leaves the row
        set unchanged rather than raising, so callers can retry safely.
    Args:
        tenant_id: Id of the tenant
        permission_name: Name of the permission

    Returns:
        None

    Raises:
        TenantNotFoundError: If the tenant does not exist.
        PermissionNotFoundError: If no permission with that name exists.
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
            raise PermissionNotFoundError(message=f"{permission_name} not found")

        await session.execute(
            delete(TenantDefaultPermissions).where(
                TenantDefaultPermissions.tenant_id == tenant.id,
                TenantDefaultPermissions.permission_id == permission_entity.id,
            )
        )

        await session.commit()
