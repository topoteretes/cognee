from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_user
from cognee.modules.users.models import (
    Role,
)
from cognee.modules.users.permissions.methods import has_user_management_permission


async def create_role(
    role_name: str,
    owner_id: UUID,
) -> UUID:
    """
        Create a new role with the given name in the request owner's current
        tenant, if they can manage users in it. The tenant owner always can.
    Args:
        role_name: Name of the new role.
        owner_id: Id of the request owner.

    Returns:
        UUID: Id of the new role.

    Raises:
        PermissionDeniedError: If the request owner cannot manage users in
            their current tenant.
        TenantNotFoundError: If their current tenant does not exist.
    """
    db_engine = get_relational_engine()

    # Resolve the user and check permission (each opens its own session) BEFORE
    # opening ours, so this request never holds two pooled connections at once:
    # that overlap deadlocks the pool under concurrency (issue #4197 class).
    user = await get_user(owner_id)
    tenant_id = user.tenant_id

    await has_user_management_permission(requester_id=owner_id, tenant_id=tenant_id)

    async with db_engine.get_async_session() as session:
        try:
            # Add association directly to the association table
            role = Role(name=role_name, tenant_id=tenant_id)
            session.add(role)
        except IntegrityError as e:
            raise EntityAlreadyExistsError(message="Role already exists for tenant.") from e

        await session.commit()
        await session.refresh(role)
        return role.id
