from fastapi_users.exceptions import UserNotExists
from cognee.infrastructure.databases.relational import get_relational_engine
from ..get_user_manager import get_user_manager_context
from ..get_user_db import get_user_db_context


async def delete_user(email: str):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.get_by_email(email)
                    await user_manager.delete(user)
    except UserNotExists as error:
        print(f"User {email} doesn't exist")
        raise error
