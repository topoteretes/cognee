from uuid import UUID

from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.capabilities.methods import require_role_capabilities
from cognee.modules.users.exceptions import (
    RoleNotFoundError,
    TenantNotFoundError,
    UserNotFoundError,
)
from cognee.modules.users.models import (
    Role,
    User,
    UserRole,
    UserTenant,
)
from cognee.modules.users.permissions.methods import has_user_management_permission

# SQLSTATE for a foreign key violation. Only Postgres enforces those here.
FOREIGN_KEY_VIOLATION = "23503"


async def add_user_to_role(user_id: UUID, role_id: UUID, owner_id: UUID):
    """
        Add a user with the given id to the role with the given id, if the
        request owner can manage users in the role's tenant and holds every
        capability the role carries. The tenant owner always can.
    Args:
        user_id: Id of the user.
        role_id: Id of the role.
        owner_id: Id of the request owner.

    Returns:
        None

    Raises:
        UserNotFoundError: If the user does not exist.
        RoleNotFoundError: If the role does not exist.
        TenantNotFoundError: If the user is not a member of the role's tenant.
        PermissionDeniedError: If the request owner cannot manage users in the
            role's tenant, or the role carries a capability they do not hold.
        EntityAlreadyExistsError: If the user already has the role.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Each lookup is checked before anything reads off it. Previously the
        # tenant query dereferenced `role.tenant_id` and the membership load
        # dereferenced `user`, both above their own guards, so an unknown role
        # or user id raised AttributeError on None instead of the
        # RoleNotFoundError / UserNotFoundError the caller (and the permissions
        # router) expects.
        user = (await session.execute(select(User).where(User.id == user_id))).scalars().first()

        if not user:
            raise UserNotFoundError

        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise RoleNotFoundError

        user_tenants = await user.awaitable_attrs.tenants

        if role.tenant_id not in [user_tenant.id for user_tenant in user_tenants]:
            raise TenantNotFoundError(
                message="User tenant does not match role tenant. User cannot be added to role."
            )

        tenant_id = role.tenant_id
        role_name = role.name

    # Both checks open their own session(s); run them OUTSIDE the session above
    # so we never hold two pooled connections at once (#4197 class).
    await has_user_management_permission(requester_id=owner_id, tenant_id=tenant_id)
    await require_role_capabilities(owner_id, role_id, tenant_id, role_name)

    # INSERT ... SELECT, so the membership is written only if, when the row goes
    # in, the role still exists and the user is still a member of its tenant.
    # Both were checked above, but in another session: a role deleted in
    # between would leave a membership pointing at nothing on SQLite, and a
    # user removed from the tenant in between would get the role's
    # capabilities back if they were ever added again. FOR SHARE on the
    # membership row makes a concurrent removal on Postgres either wait for
    # this insert or make it find no row; SQLite serializes writers anyway.
    insert_membership = insert(UserRole).from_select(
        ["user_id", "role_id"],
        select(UserTenant.user_id, Role.id)
        .join(UserTenant, UserTenant.tenant_id == Role.tenant_id)
        .where(Role.id == role_id, UserTenant.user_id == user_id)
        .with_for_update(read=True, of=UserTenant),
    )

    async with db_engine.get_async_session() as session:
        try:
            inserted = (await session.execute(insert_membership)).rowcount
        except IntegrityError as error:
            # A foreign key violation means the role or the user was deleted
            # while this ran (Postgres); anything else is the primary key.
            if getattr(error.orig, "sqlstate", None) != FOREIGN_KEY_VIOLATION:
                raise EntityAlreadyExistsError(message="User is already part of group.")
            await session.rollback()
            inserted = 0

        if inserted == 0:
            raise await _why_not_added(session, user_id, role_id)

        await session.commit()


async def _why_not_added(session, user_id: UUID, role_id: UUID) -> Exception:
    """Name what went away between the checks and the insert."""
    role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()
    if role is None:
        return RoleNotFoundError()

    user = (await session.execute(select(User.id).where(User.id == user_id))).first()
    if user is None:
        return UserNotFoundError()

    return TenantNotFoundError(
        message="User tenant does not match role tenant. User cannot be added to role."
    )
