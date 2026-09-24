from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.future import select

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import (
    Role,
    UserRole,
)
from cognee.modules.users.models.ACL import ACL
from cognee.modules.users.models.Principal import Principal
from cognee.modules.users.models.PrincipalCapability import PrincipalCapability
from cognee.modules.users.permissions.methods import has_user_management_permission


async def delete_role(role_id: UUID, owner_id: UUID):
    """
    Delete a role and its associations.

    Args:
        role_id: Id of the role to delete.
        owner_id: Id of the request owner.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise EntityNotFoundError(message="Role not found.")

        tenant_id = role.tenant_id

    # has_user_management_permission opens its own session(s); run it OUTSIDE the
    # session above so we never hold two pooled connections at once (#4197 class).
    await has_user_management_permission(requester_id=owner_id, tenant_id=tenant_id)

    async with db_engine.get_async_session() as session:
        # Lock the role before touching anything that references it. Otherwise
        # on Postgres an add_user_to_role committing between the membership
        # delete and the role delete leaves a membership behind and the role
        # delete fails its foreign key. With the lock, a membership insert
        # already in flight finishes first and is removed below, and a later
        # one waits and then finds the role gone. SQLite ignores FOR UPDATE and
        # serializes writers anyway.
        await session.execute(select(Role.id).where(Role.id == role_id).with_for_update())

        # Remove all user-role associations
        await session.execute(delete(UserRole).where(UserRole.role_id == role_id))

        # Remove all ACL entries for this role's principal
        await session.execute(delete(ACL).where(ACL.principal_id == role_id))

        # Remove the capabilities granted to the role. The CASCADE on
        # principal_capabilities is not enforced on SQLite, same as for ACL.
        await session.execute(
            delete(PrincipalCapability).where(PrincipalCapability.principal_id == role_id)
        )

        # Delete both joined-table rows so the base Principal is not orphaned.
        await session.execute(delete(Role).where(Role.id == role_id))
        await session.execute(delete(Principal).where(Principal.id == role_id))

        await session.commit()
