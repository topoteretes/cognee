from uuid import UUID

from sqlalchemy import Uuid, column, delete, table, update
from sqlalchemy.ext.asyncio import AsyncSession

# The policy table by name, not its model: the model references integration_credentials,
# so registering it is only safe where the host defines that table, and importing this
# module must register nothing. A test pins that these names exist on the model.
_SYNC_POLICIES = table(
    "integration_sync_policies",
    column("principal_id", Uuid),
    column("paused_by", Uuid),
)


async def delete_sync_policies(session: AsyncSession, principal_ids: list[UUID]) -> None:
    """Delete the sync policies of ``principal_ids``.

    Explicit because SQLite does not enforce the cascade. A policy one of them paused on
    another principal keeps its pause but loses the pauser.

    Runs on the session of the delete it belongs to, so the policies go in its transaction.
    """
    await session.execute(
        delete(_SYNC_POLICIES).where(_SYNC_POLICIES.c.principal_id.in_(principal_ids))
    )
    await session.execute(
        update(_SYNC_POLICIES)
        .where(_SYNC_POLICIES.c.paused_by.in_(principal_ids))
        .values(paused_by=None)
    )
