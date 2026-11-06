"""Read and write sync policies, scoped to the caller.

Which principal owns a connection is the host's rule (the SDK API server uses
the credential's ``workspace_id`` when set, else its ``user_id``; Cloud uses its
``tenant_id``), so these functions take the owner principal and the credential
id from the host and never read the credential row themselves.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm.exc import StaleDataError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods.get_visible_user_ids import get_visible_user_ids
from cognee.modules.users.models import Tenant, User

from .exceptions import SyncPausedError, SyncPolicyConflictError, SyncPolicyNotFoundError
from .models import IntegrationSyncPolicy
from .next_due import next_due
from .policy import SyncPolicy


def stored_policy(row: IntegrationSyncPolicy) -> SyncPolicy:
    """The row's schedule, validated again before anything computes a slot from it.

    ``next_due`` trusts its input, and a row edited by hand (interval 0, no weekday)
    would make it loop forever, so a bad row raises ``ValidationError`` here instead.
    """
    return SyncPolicy.model_validate(row, from_attributes=True)


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

    Checks that the caller may act for ``principal_id``, not that it owns ``credential_id``:
    the host resolves both ids from its own credential table. A first save that loses a race
    raises ``SyncPolicyConflictError``, one that loses to a delete ``SyncPolicyNotFoundError``.
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
        except IntegrityError as error:
            # Another first save of this row committed between the read and the insert.
            # The host passes a principal and a credential that exist, so the unique
            # index refused it, or the credential was deleted meanwhile and the retry
            # reports it missing.
            raise SyncPolicyConflictError() from error
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


async def _change_policy(
    user: User,
    principal_id: UUID,
    credential_id: UUID | None,
    change: Callable[[IntegrationSyncPolicy], None],
) -> IntegrationSyncPolicy:
    await _require_actable(user, principal_id)
    async with get_relational_engine().get_async_session() as session:
        row = (
            await session.execute(
                select(IntegrationSyncPolicy).where(*_row_filter(principal_id, credential_id))
            )
        ).scalar_one_or_none()
        if row is None:
            raise SyncPolicyNotFoundError()
        change(row)
        # Always write, as save_sync_policy does, so a delete that ran meanwhile raises.
        flag_modified(row, "paused")
        try:
            await session.commit()
        except StaleDataError as error:
            raise SyncPolicyNotFoundError() from error
    return row


async def pause_sync_policy(
    user: User,
    principal_id: UUID,
    credential_id: UUID | None = None,
    now: datetime | None = None,
) -> IntegrationSyncPolicy:
    """Hold the policy's slots; they pass without a run. A run in progress finishes.

    A paused default holds every connection of its owner, including those with their
    own row. Pausing again keeps the first ``paused_by`` and ``paused_at``.
    """
    now = now or datetime.now(timezone.utc)

    def pause(row: IntegrationSyncPolicy) -> None:
        if not row.paused:
            row.paused, row.paused_by, row.paused_at = True, user.id, now

    return await _change_policy(user, principal_id, credential_id, pause)


async def resume_sync_policy(
    user: User,
    principal_id: UUID,
    credential_id: UUID | None = None,
    now: datetime | None = None,
) -> IntegrationSyncPolicy:
    """Release a paused policy. Its next slot is the first one after now."""
    now = now or datetime.now(timezone.utc)

    def resume(row: IntegrationSyncPolicy) -> None:
        if row.paused:
            row.paused, row.paused_by, row.paused_at = False, None, None
            row.next_due_at = next_due(stored_policy(row), now)

    return await _change_policy(user, principal_id, credential_id, resume)


def policy_level(row: IntegrationSyncPolicy) -> str:
    """``default`` for an owner's default row, ``connection`` for a connection's own row."""
    return "default" if row.credential_id is None else "connection"


async def _connection_rows(
    credential_id: UUID, owner_principal_id: UUID
) -> tuple[IntegrationSyncPolicy | None, IntegrationSyncPolicy | None]:
    """The connection's own row and its owner's default row, each None when absent."""
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
    default = next((row for row in rows if row.credential_id is None), None)
    return own, default


async def get_effective_sync_policy(
    credential_id: UUID, owner_principal_id: UUID
) -> IntegrationSyncPolicy | None:
    """The connection's own row, else its owner's default, else None (no schedule).

    Not scoped to a caller: the tick reads it for every due connection. Routes
    use ``get_caller_effective_sync_policy``.
    """
    own, default = await _connection_rows(credential_id, owner_principal_id)
    return own or default


async def get_caller_effective_sync_policy(
    user: User, owner_principal_id: UUID, credential_id: UUID
) -> IntegrationSyncPolicy:
    """``get_effective_sync_policy`` for a caller who may act for the owner; none raises."""
    await _require_actable(user, owner_principal_id)
    row = await get_effective_sync_policy(credential_id, owner_principal_id)
    if row is None:
        raise SyncPolicyNotFoundError()
    return row


async def get_sync_pause(
    credential_id: UUID, owner_principal_id: UUID
) -> IntegrationSyncPolicy | None:
    """The row that holds this connection paused, or None.

    The owner's paused default comes first: it holds every connection, those with
    their own row too.
    """
    own, default = await _connection_rows(credential_id, owner_principal_id)
    return next((row for row in (default, own) if row is not None and row.paused), None)


async def require_sync_not_paused(credential_id: UUID, owner_principal_id: UUID) -> None:
    """Refuse an explicit sync of a paused connection, naming the level that holds it."""
    pause = await get_sync_pause(credential_id, owner_principal_id)
    if pause is not None:
        raise SyncPausedError(policy_level(pause))
