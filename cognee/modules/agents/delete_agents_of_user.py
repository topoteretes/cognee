from uuid import UUID

from sqlalchemy import Uuid, column, delete, select, table, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.agents.registry import _remove_from_registry
from cognee.modules.users.methods.get_agent_user_ids import get_agent_user_ids
from cognee.modules.users.models import User
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration
from cognee.modules.users.models.UserApiKey import UserApiKey

# The policy table by name, not its model: the model references integration_credentials,
# so registering it is only safe where the host defines that table, and importing this
# module must register nothing. A test pins that these names exist on the model.
_SYNC_POLICIES = table(
    "integration_sync_policies",
    column("principal_id", Uuid),
    column("paused_by", Uuid),
)


async def delete_agents_of_user(session: AsyncSession, user_id: UUID) -> None:
    """Delete the agents ``user_id`` created, and the agents those created in turn.

    Deleting rather than only revoking the keys, because an agent without a
    parent is a user nobody owns and nobody can list or delete. Datasets an
    agent owns go the way a deleted user's own datasets go: they keep the
    deleted id as their owner, and on Postgres their database mapping goes
    with it.

    Runs on the session of the delete it belongs to, so the agents and their
    parent are removed in one transaction and the delete holds one connection
    however deep the tree is. The keys, configuration rows and sync policies are
    deleted explicitly because SQLite does not enforce the cascade; a policy the
    deleted user paused on another principal keeps its pause but loses the pauser.
    """
    agent_ids = await get_agent_user_ids(session, user_id)
    await session.execute(delete(UserApiKey).where(UserApiKey.user_id.in_(agent_ids)))
    await session.execute(
        delete(PrincipalConfiguration).where(PrincipalConfiguration.owner_id.in_(agent_ids))
    )
    gone_ids = [user_id, *agent_ids]
    await session.execute(delete(_SYNC_POLICIES).where(_SYNC_POLICIES.c.principal_id.in_(gone_ids)))
    await session.execute(
        update(_SYNC_POLICIES)
        .where(_SYNC_POLICIES.c.paused_by.in_(gone_ids))
        .values(paused_by=None)
    )
    for agent in (await session.execute(select(User).where(User.id.in_(agent_ids)))).scalars():
        await session.delete(agent)
        _remove_from_registry(agent.id)
