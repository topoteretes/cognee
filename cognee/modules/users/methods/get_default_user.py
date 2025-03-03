from types import SimpleNamespace

from sqlalchemy.orm import selectinload
from sqlalchemy.future import select
from cognee.modules.users.models import User, Tenant
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.create_default_user import create_default_user


async def get_default_user():
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.email == "default_user@example.com")
        )

        result = await session.execute(query)
        user = result.scalars().first()

        if user is None:
            return await create_default_user()

        # Get tenant from user
        result = await session.execute(select(Tenant).where(Tenant.id == user.tenant_id))
        tenant = result.scalars().first()

        # We return a SimpleNamespace to have the same user type as our SaaS
        # SimpleNamespace is just a dictionary which can be accessed through attributes
        ret_val = SimpleNamespace(id=user.id, tenant=tenant.name, roles=[])
        return ret_val
