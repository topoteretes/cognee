import hashlib
# from cognee.infrastructure.databases.relational import get_relational_engine
from .create_user import create_user

async def create_default_user():
    default_user_email = "default_user@example.com"
    default_user_password = "default_password"

    user = await create_user(
        email = default_user_email,
        password = await hash_password(default_user_password),
        is_superuser = True,
        is_active = True,
        is_verified = True,
    )

    # db_engine = get_relational_engine()
    # async with db_engine.get_async_session() as session:
    #     await session.refresh(user)

    return user

async def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()
