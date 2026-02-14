from sqlalchemy import select

from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.Tenant import Tenant
from cognee.modules.users.models.UserTenant import UserTenant


async def get_user_tenants(user: User):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        tenant_results = await session.execute(
            select(Tenant)
            .join(UserTenant, Tenant.id == UserTenant.tenant_id)
            .where(UserTenant.user_id == user.id)
        )
        tenants = tenant_results.scalars().all()

        # Format response
        tenant_list = []
        for tenant in tenants:
            tenant_list.append(
                {
                    "id": str(tenant.id),
                    "name": tenant.name,
                }
            )

    return tenant_list
