"""Read and write sync policies, scoped to the caller.

Which principal owns a connection is the host's rule (the SDK API server uses
the credential's ``workspace_id`` when set, else its ``user_id``; Cloud uses its
``tenant_id``), so these functions take the owner principal and the credential
id from the host and never read the credential row themselves.
"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm.exc import StaleDataError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.get_visible_user_ids import get_visible_user_ids
from cognee.modules.users.models import Tenant, User

from .exceptions import SyncPolicyNotFoundError
from .models import IntegrationSyncPolicy
from .next_due import next_due
from .policy import SyncPolicy


async def _actable_principal_ids(user: User) -> list[UUID]:
    """The caller, the agent users it created, and the tenants it owns.

    Downward only, as ``get_principal_configuration``: an agent cannot act on its
    parent's policies, and a tenant member who does not own the tenant can neither
    read nor change the tenant's default.
    """
    ids = await get_visible_user_ids(user.id)
    async with get_relational_engine().get_async_session() as session:
        owned = (await session.execute(select(Tenant.id).where(Tenant.owner_id == user.id))).all()
    return [*ids, *(row.id for row in owned)]


def _row_filter(principal_id: UUID, credential_id: UUID | None):
    credential = (
        IntegrationSyncPolicy.credential_id.is_(None)
        if credential_id is None
        else IntegrationSyncPolicy.credential_id == credential_id
    )
    return IntegrationSyncPolicy.principal_id == principal_id, credential


async def _require_actable(user: User, principal_id: UUID) -> None:
    # Another owner's policy raises exactly like a missing one.
    if principal_id not in await _actable_principal_ids(user):
        raise SyncPolicyNotFoundError()


async def get_sync_policy(
    user: User, principal_id: UUID, credential_id: UUID | None = None
) -> IntegrationSyncPolicy:
    """The principal's default (``credential_id`` None) or one connection's own row."""
    await _require_actable(user, principal_id)
    async with get_relational_engine().get_async_session() as session:
        row = (
            await session.execute(
                select(IntegrationSyncPolicy).where(*_row_filter(principal_id, credential_id))
            )
        ).scalar_one_or_none()
    if row is None:
        raise SyncPolicyNotFoundError()
    return row


async def save_sync_policy(
    user: User,
    principal_id: UUID,
    policy: SyncPolicy,
    credential_id: UUID | None = None,
    now: datetime | None = None,
) -> IntegrationSyncPolicy:
    """Create or replace a policy and compute its next slot. A pause is kept as it was.

    Checks that the caller may act for ``principal_id``, not that ``credential_id``
    belongs to it: the credential table and its owner column differ per host, so
    the host resolves both ids and passes a connection that principal owns.

    Two first saves of the same row at once: the second insert hits the unique
    index and raises, it does not overwrite. A save that loses a race with a delete
    of the same row raises ``SyncPolicyNotFoundError``.
    """
    await _require_actable(user, principal_id)
    now = now or datetime.now(timezone.utc)
    values = {
        "interval_minutes": policy.interval_minutes,
        "weekdays": list(policy.weekdays),
        "timezone": policy.timezone,
        "next_due_at": next_due(policy, now),
    }
    async with get_relational_engine().get_async_session() as session:
        row = (
            await session.execute(
                select(IntegrationSyncPolicy).where(*_row_filter(principal_id, credential_id))
            )
        ).scalar_one_or_none()
        if row is None:
            row = IntegrationSyncPolicy(
                principal_id=principal_id, credential_id=credential_id, **values
            )
            session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
            # Always write: an identical policy changes no attribute, and without an UPDATE
            # a row deleted meanwhile goes unnoticed and the save reports success.
            flag_modified(row, "interval_minutes")
        try:
            await session.commit()
        except StaleDataError as error:
            # The row was deleted between the read and the update.
            raise SyncPolicyNotFoundError() from error
    # Nothing is reloaded: a reload fails when another connection deletes the row after the
    # commit. The row keeps what was written (expire_on_commit=False); on a new row
    # paused_by and paused_at were never set and read as None. SQLite returns datetimes it loaded
    # without their offset, so there one row can hold aware and naive values.
    return row


async def delete_sync_policy(
    user: User, principal_id: UUID, credential_id: UUID | None = None
) -> None:
    """Remove a policy. A connection without its own row inherits the default again."""
    await _require_actable(user, principal_id)
    async with get_relational_engine().get_async_session() as session:
        result = await session.execute(
            delete(IntegrationSyncPolicy).where(*_row_filter(principal_id, credential_id))
        )
        await session.commit()
    if result.rowcount == 0:
        raise SyncPolicyNotFoundError()


async def get_effective_sync_policy(
    credential_id: UUID, owner_principal_id: UUID
) -> IntegrationSyncPolicy | None:
    """The connection's own row, else its owner's default, else None (no schedule).

    Not scoped to a caller: the tick reads it for every due connection. Routes
    use ``get_sync_policy``.
    """
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(IntegrationSyncPolicy).where(
                        IntegrationSyncPolicy.principal_id == owner_principal_id,
                        (IntegrationSyncPolicy.credential_id == credential_id)
                        | IntegrationSyncPolicy.credential_id.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
    own = next((row for row in rows if row.credential_id == credential_id), None)
    return own or next(iter(rows), None)
