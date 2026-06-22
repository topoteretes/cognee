from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.agents.list_agents import AgentInfo
from cognee.modules.users.methods.get_user import get_user
from cognee.modules.users.models.UserApiKey import UserApiKey
from sqlalchemy import select


async def get_agent(agent_id: UUID, owner_id: UUID) -> AgentInfo:
    agent_user = await get_user(agent_id)

    if agent_user is None:
        raise LookupError("Agent not found")

    if agent_user.parent_user_id != owner_id:
        raise PermissionError("Not authorized to view this agent")

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        key_result = await session.execute(
            select(UserApiKey.label).where(UserApiKey.user_id == agent_id)
        )
        row = key_result.first()

    return AgentInfo(user=agent_user, api_key_label=row.label if row else None)
