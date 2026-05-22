from cognee.modules.users.methods.create_user import create_user
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.models.User import User
from cognee.infrastructure.databases.relational import get_relational_engine


async def create_agent(name: str, parent_user: User) -> tuple[User, str]:
    sanitized_name = name.lower().replace(" ", "-")
    internal_email = f"{sanitized_name}+{parent_user.id}@cognee.agent"

    agent_user = await create_user(
        email=internal_email,
        password="!",
        parent_user_id=parent_user.id,
    )

    # Lock the password — "!" is not a valid hash so pwdlib raises
    # UnknownHashError on verify, which UserManager.authenticate catches.
    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        agent_user.hashed_password = "!"
        session.add(agent_user)
        await session.commit()

    if parent_user.tenant_id is not None:
        async with relational_engine.get_async_session() as session:
            agent_user.tenant_id = parent_user.tenant_id
            session.add(agent_user)
            await session.commit()

    agent_api_key = await create_api_key(agent_user, sanitized_name)
    return agent_user, agent_api_key.api_key
