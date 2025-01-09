from fastapi_users.exceptions import UserAlreadyExists
from cognee.infrastructure.databases.relational import get_relational_engine
from ..get_user_manager import get_user_manager_context
from ..get_user_db import get_user_db_context
from ..models.User import UserCreate


async def create_user(
    email: str,
    password: str,
    is_superuser: bool = False,
    is_active: bool = True,
    is_verified: bool = False,
    auto_login: bool = False,
):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.create(
                        UserCreate(
                            email=email,
                            password=password,
                            is_superuser=is_superuser,
                            is_active=is_active,
                            is_verified=is_verified,
                        )
                    )

                    if auto_login:
                        await session.refresh(user)

                    return user
    except UserAlreadyExists as error:
        print(f"User {email} already exists")
        raise error
