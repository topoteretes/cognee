from uuid import UUID

from cognee.modules.agents.registry import delete_user_agent_connections
from cognee.modules.users.methods.delete_user import delete_user
from cognee.modules.users.methods.get_user import get_user


async def delete_agent(agent_id: UUID, owner_id: UUID) -> None:
    agent_user = await get_user(agent_id)

    if agent_user is None:
        raise LookupError("Agent not found")

    if agent_user.parent_user_id != owner_id:
        raise PermissionError("Not authorized to delete this agent")

    await delete_user_agent_connections(agent_id)
    await delete_user(agent_user.email)
