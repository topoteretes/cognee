from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.get_user_agents import get_user_agents
from cognee.modules.users.models.User import User
from cognee.modules.users.models.UserApiKey import UserApiKey


@dataclass
class AgentInfo:
    user: User
    api_key_label: str | None


async def list_agents(owner_id: UUID) -> list[AgentInfo]:
    agents = await get_user_agents(owner_id)
    agent_ids = [a.id for a in agents]
    if not agent_ids:
        return []

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        key_result = await session.execute(
            select(UserApiKey.user_id, UserApiKey.label).where(UserApiKey.user_id.in_(agent_ids))
        )
        label_by_user = {row.user_id: row.label for row in key_result}

        return [
            AgentInfo(user=agent, api_key_label=label_by_user.get(agent.id)) for agent in agents
        ]
