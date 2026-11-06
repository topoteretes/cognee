"""Claim due sync slots and start the connections they cover.

The claim starts a slot once however many processes tick. Two runs of one
connection are kept apart only by the host's in-process guard, so a host runs
the tick, manual syncs and webhooks in one process per database.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import IntegrationSyncPolicy
from .next_due import next_due
from .store import get_effective_sync_policy, get_sync_pause_level, stored_policy

logger = get_logger("sync_policy_tick")

# The host's two parts: a principal's active connections, and starting one sync
# without waiting for it. The SDK API server and Cloud keep different credential
# tables and start syncs differently.
OwnedCredentialIds = Callable[[UUID], Awaitable[list[UUID]]]
StartSync = Callable[[UUID], Awaitable[None]]


async def claim_due_slot(row: IntegrationSyncPolicy, now: datetime) -> bool:
    """Move the row to its next slot; True only for the one caller that moved it.

    It matches only while the row still holds the slot this caller read and is not
    paused, so one of several workers wins, with no row lock or ``SKIP LOCKED``
    (SQLite has neither). The next slot is the first after ``now``: missed slots are one run.
    """
    upcoming = next_due(stored_policy(row), now)
    async with get_relational_engine().get_async_session() as session:
        result = await session.execute(
            update(IntegrationSyncPolicy)
            .where(
                IntegrationSyncPolicy.id == row.id,
                IntegrationSyncPolicy.next_due_at == row.next_due_at,
                ~IntegrationSyncPolicy.paused,
            )
            .values(next_due_at=upcoming)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return result.rowcount == 1


async def _due_rows(now: datetime) -> list[IntegrationSyncPolicy]:
    # NOT paused here as well as in the claim: a paused row's slot never moves, so this
    # keeps it out of every scan, and it is the partial index ix_integration_sync_policies_due.
    async with get_relational_engine().get_async_session() as session:
        return list(
            (
                await session.execute(
                    select(IntegrationSyncPolicy).where(
                        IntegrationSyncPolicy.next_due_at <= now, ~IntegrationSyncPolicy.paused
                    )
                )
            )
            .scalars()
            .all()
        )


async def _connections_of(
    row: IntegrationSyncPolicy, owned_credential_ids: OwnedCredentialIds
) -> list[UUID]:
    """The connections a claimed slot starts."""
    if row.credential_id is not None:
        # The row's principal must still own the connection: a reconnect can hand the
        # same credential row to another owner, whose own policies then apply.
        if row.credential_id not in await owned_credential_ids(row.principal_id):
            return []
        return [row.credential_id]
    # A default row starts the owner's connections that inherit it: not those with a
    # row of their own, which schedules or pauses them.
    connections = []
    for credential_id in await owned_credential_ids(row.principal_id):
        effective = await get_effective_sync_policy(credential_id, row.principal_id)
        if effective is not None and effective.id == row.id:
            connections.append(credential_id)
    return connections


async def run_due_syncs(
    now: datetime, owned_credential_ids: OwnedCredentialIds, start: StartSync
) -> int:
    """Claim every due slot and start each connection it covers. Returns how many started.

    ``start`` hands a sync off and returns; a sync that fails later is the host's to
    report, so one connection does not stop the others.
    """
    started = 0
    for row in await _due_rows(now):
        # Read before the claim: a read that fails here leaves the slot due for the
        # next tick instead of claimed with nothing started.
        connections = await _connections_of(row, owned_credential_ids)
        try:
            claimed = await claim_due_slot(row, now)
        except ValidationError as error:
            # Only a row edited outside the store fails here. It stays due and is
            # reported on every tick until it is saved again.
            logger.error("Sync policy %s is invalid and is not scheduled: %s", row.id, error)
            continue
        if not claimed:
            continue
        # A paused default holds a connection with its own row too: the slot passes
        # without a run. Read after the claim, so a pause that lands meanwhile counts;
        # a read that fails here loses this one slot, as a crash after the claim would.
        if (
            connections
            and row.credential_id is not None
            and await get_sync_pause_level(row.credential_id, row.principal_id) is not None
        ):
            continue
        for credential_id in connections:
            await start(credential_id)
            started += 1
    return started
