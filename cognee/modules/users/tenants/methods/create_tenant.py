from uuid import UUID
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Tenant
from cognee.modules.users.methods import get_user


async def create_tenant(tenant_name: str, user_id: UUID) -> UUID:
    """
        Create a new tenant with the given name, for the user with the given id.
        This user is the owner of the tenant.
    Args:
        tenant_name: Name of the new tenant.
        user_id: Id of the user.

    Returns:
        None
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            user = await get_user(user_id)
            if user.tenant_id:
                raise EntityAlreadyExistsError(
                    message="User already has a tenant. New tenant cannot be created."
                )

            tenant = Tenant(name=tenant_name, owner_id=user_id)
            session.add(tenant)
            await session.flush()

            user.tenant_id = tenant.id
            await session.merge(user)
            await session.commit()
            return tenant.id
        except IntegrityError as e:
            raise EntityAlreadyExistsError(message="Tenant already exists.") from e
