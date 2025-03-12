from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Tenant


async def create_tenant(tenant_name: str):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            # Add association directly to the association table
            tenant = Tenant(name=tenant_name)
            session.add(tenant)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Tenant already exists.")

        await session.commit()
        await session.refresh(tenant)
