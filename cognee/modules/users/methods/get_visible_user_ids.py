"""Resolve which user IDs a caller's queries should be scoped to.

Shared across every module that needs to answer "this caller, plus
anything delegated to them" — sessions, agent connections, usage
stats, cost aggregates, ... Previously duplicated near-verbatim in
``cognee.modules.agents.operations`` and
``cognee.modules.session_lifecycle`` before being consolidated here.
"""

from uuid import UUID as UUIDType

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User


async def _child_agent_user_ids(user_id: UUIDType) -> list[UUIDType]:
    """Return user IDs of agents whose ``parent_user_id`` matches this user."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (await session.execute(select(User.id).where(User.parent_user_id == user_id))).all()
        return [row.id for row in rows]


async def get_visible_user_ids(user_id: UUIDType) -> list[UUIDType]:
    """``user_id`` plus any of its child agent IDs.

    Takes a bare ID rather than a ``User`` object — every call site
    only ever needs ``.id``, and this keeps the function usable by
    callers that work with plain IDs (e.g. cache/session lookups) as
    well as ones with a resolved ``User``.
    """
    ids = [user_id]
    ids.extend(await _child_agent_user_ids(user_id))
    return ids
