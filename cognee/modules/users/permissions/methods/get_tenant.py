import sqlalchemy.exc
from sqlalchemy import select
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import TenantNotFoundError
from ...models.Tenant import Tenant


async def get_tenant(tenant_id: UUID):
    """
        Return information about the tenant based on the given id.
    Args:
        tenant_id: Id of the given tenant

    Returns
        Information about the given tenant.

    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        try:
            result = await session.execute(select(Tenant).where(Tenant.id == tenant_id))
            tenant = result.unique().scalar_one()
            if not tenant:
                raise TenantNotFoundError
            return tenant
        except sqlalchemy.exc.NoResultFound:
            raise TenantNotFoundError(message=f"Could not find tenant: {tenant_id}")
