from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.users.permissions.methods import get_tenant
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import (
    Role,
)


async def create_role(
    role_name: str,
    owner_id: UUID,
) -> UUID:
    """
        Create a new role with the given name, if the request owner with the given id
        has the necessary permission.
    Args:
        role_name: Name of the new role.
        owner_id: Id of the request owner.

    Returns:
        None
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = await get_user(owner_id)
        tenant = await get_tenant(user.tenant_id)

        if owner_id != tenant.owner_id:
            raise PermissionDeniedError(
                "User submitting request does not have permission to create role for tenant."
            )

        try:
            # Add association directly to the association table
            role = Role(name=role_name, tenant_id=tenant.id)
            session.add(role)
        except IntegrityError as e:
            raise EntityAlreadyExistsError(message="Role already exists for tenant.") from e

        await session.commit()
        await session.refresh(role)
        return role.id
