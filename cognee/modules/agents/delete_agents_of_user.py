from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.agents.registry import _remove_from_registry
from cognee.modules.users.methods.get_agent_user_ids import get_agent_user_ids
from cognee.modules.users.models import User
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration
from cognee.modules.users.models.UserApiKey import UserApiKey


async def delete_agents_of_user(session: AsyncSession, user_id: UUID) -> None:
    """Delete the agents ``user_id`` created, and the agents those created in turn.

    Deleting rather than only revoking the keys, because an agent without a
    parent is a user nobody owns and nobody can list or delete. Datasets an
    agent owns go the way a deleted user's own datasets go: they keep the
    deleted id as their owner, and on Postgres their database mapping goes
    with it.

    Runs on the session of the delete it belongs to, so the agents and their
    parent are removed in one transaction and the delete holds one connection
    however deep the tree is. The keys and configuration rows are deleted
    explicitly because SQLite does not enforce the cascade.
    """
    agent_ids = await get_agent_user_ids(session, user_id)
    await session.execute(delete(UserApiKey).where(UserApiKey.user_id.in_(agent_ids)))
    await session.execute(
        delete(PrincipalConfiguration).where(PrincipalConfiguration.owner_id.in_(agent_ids))
    )
    for agent in (await session.execute(select(User).where(User.id.in_(agent_ids)))).scalars():
        await session.delete(agent)
        _remove_from_registry(agent.id)
