"""Caller-visibility helpers shared across HTTP surfaces.

Who a caller "is" for read purposes: their own user id plus any child
agent sub-users (plugins provisioned with their own identity). The
user-id scoping itself lives in the consolidated
``cognee.modules.users.methods.get_visible_user_ids``; this module keeps
the ``User``-object wrapper used by the integrations surfaces and the
agent-email lookup used for attribution.
"""

from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_visible_user_ids
from cognee.modules.users.models import User


async def child_agent_emails(user_id: UUID) -> dict[UUID, str]:
    """Return ``{agent user id: internal email}`` of this user's child agents.

    The email is the deterministic identity ``create_agent`` derives
    (``<name>+<parent_id>@cognee.agent``) — server-assigned, unlike the
    agent-writable principal-configuration blob, so callers can trust it
    for attribution.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (
            await session.execute(select(User.id, User.email).where(User.parent_user_id == user_id))
        ).all()
        return {row.id: row.email for row in rows}


async def visible_user_ids(user: User) -> list[UUID]:
    """User's own ID plus any child agent IDs."""
    return await get_visible_user_ids(user.id)
