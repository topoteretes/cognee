"""Caller-visibility helpers shared across HTTP surfaces.

Who a caller "is" for read purposes: their own user id plus any child
agent sub-users (plugins provisioned with their own identity), and the
datasets they hold read permission on. Lifted out of the sessions router
so other surfaces (e.g. the integrations status endpoint) can reuse the
same rule without importing router privates.
"""

from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)


async def permitted_dataset_ids_for(user: User) -> list[UUID]:
    """Return the UUIDs of datasets this user can read (empty on none)."""
    try:
        datasets = await get_specific_user_permission_datasets(user.id, "read", None)
        return [ds.id for ds in datasets] if datasets else []
    except PermissionDeniedError:
        return []
    except Exception:
        return []


async def child_agent_user_ids(user_id: UUID) -> list[UUID]:
    """Return user IDs of agents whose parent_user_id matches this user."""
    return list(await child_agent_emails(user_id))


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
    ids = [user.id]
    ids.extend(await child_agent_user_ids(user.id))
    return ids
