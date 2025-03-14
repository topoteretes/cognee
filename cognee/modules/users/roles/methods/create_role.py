from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import (
    Role,
)


async def create_role(
    role_name: str,
    tenant_id: UUID,
):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            # Add association directly to the association table
            role = Role(name=role_name, tenant_id=tenant_id)
            session.add(role)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Role already exists for tenant.")

        await session.commit()
        await session.refresh(role)
