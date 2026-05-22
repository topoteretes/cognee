from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.methods import create_user
from cognee.modules.users.models.User import User
from sqlalchemy import update


async def create_agent(name: str, parent_user: User) -> tuple[User, str]:
    sanitized_name = name.lower().replace(" ", "-")
    internal_email = f"{sanitized_name}+{parent_user.id}@cognee.agent"

    agent_user = await create_user(
        email=internal_email,
        password="!",
        parent_user_id=parent_user.id,
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        values = {"hashed_password": "!"}
        if parent_user.tenant_id is not None:
            values["tenant_id"] = parent_user.tenant_id
        await session.execute(update(User).where(User.id == agent_user.id).values(**values))
        await session.commit()

    agent_api_key = await create_api_key(agent_user, sanitized_name)

    return agent_user, agent_api_key.api_key
