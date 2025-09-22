from uuid import UUID

from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    TenantNotFoundError,
    PermissionDeniedError,
)
from cognee.modules.users.models import (
    User,
    Role,
    Tenant,
    UserRole,
)


async def add_user_to_role(user_id: UUID, role_id: UUID, owner_id: UUID):
    """
        Add a user with the given id to the role with the given id.
    Args:
        user_id: Id of the user.
        role_id: Id of the role.
        owner_id: Id of the request owner.

    Returns:
        None

    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalars().first()
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()
        tenant = (
            (await session.execute(select(Tenant).where(Tenant.id == role.tenant_id)))
            .scalars()
            .first()
        )

        if not user:
            raise UserNotFoundError
        elif not role:
            raise RoleNotFoundError
        elif user.tenant_id != role.tenant_id:
            raise TenantNotFoundError(
                message="User tenant does not match role tenant. User cannot be added to role."
            )
        elif tenant.owner_id != owner_id:
            raise PermissionDeniedError(
                message="User submitting request does not have permission to add user to role."
            )

        try:
            # Add association directly to the association table
            create_user_role_statement = insert(UserRole).values(user_id=user_id, role_id=role_id)
            await session.execute(create_user_role_statement)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User is already part of group.")

        await session.commit()
