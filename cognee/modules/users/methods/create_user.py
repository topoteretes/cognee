from uuid import UUID

from fastapi_users.exceptions import UserAlreadyExists

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.get_user_db import get_user_db_context
from cognee.modules.users.get_user_manager import get_user_manager_context
from cognee.modules.users.models.User import UserCreate

# Stored as hashed_password for an account with NO password. Not a valid hash by
# construction (Unix "locked account" convention), so pwdlib raises
# UnknownHashError on verify and the login path refuses it.
NO_PASSWORD_SENTINEL = "!"


async def create_user(
    email: str,
    password: str | None,
    is_superuser: bool = False,
    is_active: bool = True,
    is_verified: bool = False,
    auto_login: bool = False,
    parent_user_id: UUID | None = None,
):
    try:
        relational_engine = get_relational_engine()

        async with (
            relational_engine.get_async_session() as session,
            get_user_db_context(session) as user_db,
            get_user_manager_context(user_db) as user_manager,
        ):
            if password is None:
                # A password-less account: nothing to log in with. Same row the
                # manager would build, minus a real hash -- the sentinel is not a
                # valid hash, so verification raises UnknownHashError and login is
                # refused with "This user does not have a password".
                if await user_db.get_by_email(email) is not None:
                    raise UserAlreadyExists()
                user = await user_db.create(
                    {
                        "email": email,
                        "hashed_password": NO_PASSWORD_SENTINEL,
                        "is_superuser": is_superuser,
                        "is_active": is_active,
                        "is_verified": is_verified,
                        "parent_user_id": parent_user_id,
                    }
                )
            else:
                user = await user_manager.create(
                    UserCreate(
                        email=email,
                        password=password,
                        is_superuser=is_superuser,
                        is_active=is_active,
                        is_verified=is_verified,
                        parent_user_id=parent_user_id,
                    )
                )

            if auto_login:
                await session.refresh(user)

            # Update tenants and roles information for User object
            _ = await user.awaitable_attrs.tenants
            _ = await user.awaitable_attrs.roles

            return user
    except UserAlreadyExists:
        print("A user with this email already exists")
        raise
