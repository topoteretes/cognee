"""Sync policies are scoped to the caller and inherit the owner's default (SDK-1434).

Runs on a real SQLite file with the real ``get_visible_user_ids`` (the ``database``
fixture in ``conftest.py``), as the principal-configuration scope tests do (SDK-803).
"""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

# Registers integration_credentials, the table the policy table's foreign key points at.
import cognee.modules.integrations.models
from cognee.modules.integrations.sync_policy import (
    SyncPausedError,
    SyncPolicy,
    SyncPolicyConflictError,
    SyncPolicyNotFoundError,
)
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy
from cognee.modules.users.models import Tenant, User

store = importlib.import_module("cognee.modules.integrations.sync_policy.store")

NOW = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
DAILY = SyncPolicy(interval_minutes=1440, weekdays=[0, 1, 2, 3, 4], timezone="Europe/Belgrade")
HOURLY = SyncPolicy(interval_minutes=60)


def _user():
    """A ``User`` caller, not a stand-in: a scope that read any attribute beyond the id
    (``parent_user_id``) must be judged on the real model."""
    return User(id=uuid4(), parent_user_id=None)


async def _add_agent(database, parent):
    """A real agent user row: its ``parent_user_id`` is what get_visible_user_ids reads."""
    agent_id = uuid4()
    agent = User(
        id=agent_id,
        email=f"{agent_id}@example.com",
        hashed_password="x",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        parent_user_id=parent.id,
    )
    async with database.get_async_session() as session:
        session.add(agent)
        await session.commit()
    return agent


async def _add_tenant(database, owner):
    tenant_id = uuid4()
    async with database.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme", owner_id=owner.id))
        await session.commit()
    return tenant_id


async def _count(database, **filters):
    async with database.get_async_session() as session:
        query = select(func.count()).select_from(IntegrationSyncPolicy)
        for column, value in filters.items():
            query = query.where(getattr(IntegrationSyncPolicy, column) == value)
        return (await session.execute(query)).scalar_one()


def _insert_default_row(database, principal_id: UUID):
    database.execute(
        "INSERT INTO integration_sync_policies (id, principal_id, interval_minutes, weekdays,"
        " timezone, paused, created_at, updated_at) VALUES (?, ?, 60, '[0]', 'UTC', 0,"
        " '2026-11-05 00:00:00', '2026-11-05 00:00:00')",
        (uuid4().hex, principal_id.hex),
    )


@pytest.mark.asyncio
async def test_save_computes_the_next_slot_in_utc(database):
    user = _user()

    row = await store.save_sync_policy(user, user.id, DAILY, now=NOW)

    # Thursday 15:00 Belgrade: Friday 10:00 local is 09:00 UTC.
    assert row.next_due_at.replace(tzinfo=timezone.utc) == datetime(
        2026, 11, 6, 9, 0, tzinfo=timezone.utc
    )
    # The column's type, as a read returns it, not the policy's tuple.
    assert row.weekdays == [0, 1, 2, 3, 4]
    assert (await store.get_sync_policy(user, user.id)).interval_minutes == 1440


@pytest.mark.asyncio
async def test_saving_again_replaces_the_row_and_keeps_a_pause(database):
    user, credential_id = _user(), uuid4()
    first = await store.save_sync_policy(user, user.id, DAILY, credential_id, now=NOW)
    async with database.get_async_session() as session:
        (await session.get(IntegrationSyncPolicy, first.id)).paused = True
        await session.commit()

    second = await store.save_sync_policy(user, user.id, HOURLY, credential_id, now=NOW)

    assert second.id == first.id
    assert second.interval_minutes == 60
    assert second.paused is True


@pytest.mark.asyncio
async def test_another_users_policy_reads_like_a_missing_one(database):
    owner, stranger = _user(), _user()
    await store.save_sync_policy(owner, owner.id, DAILY, now=NOW)

    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_sync_policy(stranger, owner.id)
    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_sync_policy(owner, owner.id, credential_id=uuid4())
    with pytest.raises(SyncPolicyNotFoundError):
        await store.save_sync_policy(stranger, owner.id, HOURLY, now=NOW)
    with pytest.raises(SyncPolicyNotFoundError):
        await store.delete_sync_policy(stranger, owner.id)


@pytest.mark.asyncio
async def test_parent_acts_on_its_agent_but_not_the_other_way(database):
    parent = _user()
    agent = await _add_agent(database, parent)
    await store.save_sync_policy(agent, agent.id, HOURLY, now=NOW)
    await store.save_sync_policy(parent, parent.id, DAILY, now=NOW)

    assert (await store.get_sync_policy(parent, agent.id)).interval_minutes == 60
    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_sync_policy(agent, parent.id)


@pytest.mark.asyncio
async def test_only_the_tenant_owner_reads_or_sets_the_tenant_default(database):
    owner, member = _user(), _user()
    tenant_id = await _add_tenant(database, owner)

    await store.save_sync_policy(owner, tenant_id, DAILY, now=NOW)

    assert (await store.get_sync_policy(owner, tenant_id)).principal_id == tenant_id
    with pytest.raises(SyncPolicyNotFoundError):
        await store.save_sync_policy(member, tenant_id, HOURLY, now=NOW)
    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_sync_policy(member, tenant_id)


@pytest.mark.asyncio
async def test_effective_policy_is_own_row_then_default_then_none(database):
    owner = _user()
    tenant_id = await _add_tenant(database, owner)
    own_credential, inheriting_credential = uuid4(), uuid4()

    effective = store.get_caller_effective_sync_policy
    with pytest.raises(SyncPolicyNotFoundError):
        await effective(owner, tenant_id, inheriting_credential)

    await store.save_sync_policy(owner, tenant_id, DAILY, now=NOW)
    await store.save_sync_policy(owner, tenant_id, HOURLY, own_credential, now=NOW)

    own, _ = await effective(owner, tenant_id, own_credential)
    inherited, _ = await effective(owner, tenant_id, inheriting_credential)
    assert own.interval_minutes == 60
    assert inherited.interval_minutes == 1440
    assert inherited.credential_id is None


@pytest.mark.asyncio
async def test_deleting_a_connection_row_returns_it_to_the_default(database):
    user, credential_id = _user(), uuid4()
    await store.save_sync_policy(user, user.id, DAILY, now=NOW)
    await store.save_sync_policy(user, user.id, HOURLY, credential_id, now=NOW)

    await store.delete_sync_policy(user, user.id, credential_id)

    row, _ = await store.get_caller_effective_sync_policy(user, user.id, credential_id)
    assert row.credential_id is None
    with pytest.raises(SyncPolicyNotFoundError):
        await store.delete_sync_policy(user, user.id, credential_id)


@pytest.mark.asyncio
async def test_a_principal_has_at_most_one_default_row_and_one_row_per_connection(database):
    principal_id, credential_id = uuid4(), uuid4()

    def _row(credential):
        return IntegrationSyncPolicy(
            principal_id=principal_id,
            credential_id=credential,
            interval_minutes=60,
            weekdays=[0],
            timezone="UTC",
        )

    for credential in (None, credential_id):
        async with database.get_async_session() as session:
            session.add_all([_row(credential), _row(credential)])
            with pytest.raises(IntegrityError):
                await session.commit()


@pytest.mark.asyncio
async def test_a_save_that_loses_a_race_with_a_delete_raises_not_found(database):
    user = _user()
    await store.save_sync_policy(user, user.id, DAILY, now=NOW)
    # After the save has read the row, before it writes: another connection deletes it.
    database.before_commit = lambda: database.execute("DELETE FROM integration_sync_policies")

    with pytest.raises(SyncPolicyNotFoundError):
        await store.save_sync_policy(user, user.id, HOURLY, now=NOW)

    assert await _count(database) == 0


@pytest.mark.asyncio
async def test_an_identical_save_that_loses_a_race_with_a_delete_raises_not_found(
    database, monkeypatch
):
    """Postgres reads the slot back as written, so saving the same policy again changes no
    attribute. SQLite drops the offset on read; a naive slot makes it compare equal too."""
    user = _user()
    monkeypatch.setattr(store, "next_due", lambda _policy, _after: NOW.replace(tzinfo=None))
    await store.save_sync_policy(user, user.id, DAILY, now=NOW)
    database.before_commit = lambda: database.execute("DELETE FROM integration_sync_policies")

    with pytest.raises(SyncPolicyNotFoundError):
        await store.save_sync_policy(user, user.id, DAILY, now=NOW)

    assert await _count(database) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("first_save", [True, False])
async def test_a_delete_right_after_the_commit_does_not_fail_the_save(database, first_save):
    user = _user()
    if not first_save:
        await store.save_sync_policy(user, user.id, DAILY, now=NOW)
    # The save has written; another connection deletes the row before the save returns.
    database.after_commit = lambda: database.execute("DELETE FROM integration_sync_policies")

    row = await store.save_sync_policy(user, user.id, HOURLY, now=NOW)

    assert row.interval_minutes == 60
    assert row.paused is False
    assert row.updated_at is not None
    assert await _count(database) == 0


@pytest.mark.asyncio
async def test_a_first_save_that_loses_a_race_is_a_conflict_and_does_not_overwrite(database):
    """The route answers 409, not the 500 a raw ``IntegrityError`` becomes."""
    user = _user()
    # After the save has found no row, before it inserts: another connection inserts one.
    database.before_commit = lambda: _insert_default_row(database, user.id)

    with pytest.raises(SyncPolicyConflictError):
        await store.save_sync_policy(user, user.id, DAILY, now=NOW)

    # The competing row is the one that stays, untouched.
    assert await _count(database, principal_id=user.id, interval_minutes=60) == 1
    assert await _count(database) == 1


@pytest.mark.asyncio
async def test_pause_records_who_and_when_and_a_second_pause_keeps_them(database):
    owner = _user()
    agent = await _add_agent(database, owner)
    await store.save_sync_policy(owner, agent.id, HOURLY, now=NOW)

    first = await store.pause_sync_policy(owner, agent.id, now=NOW)
    again = await store.pause_sync_policy(
        agent, agent.id, now=datetime(2026, 11, 6, tzinfo=timezone.utc)
    )

    assert first.paused is True
    assert again.paused_by == owner.id
    assert again.paused_at.replace(tzinfo=timezone.utc) == NOW


@pytest.mark.asyncio
async def test_resume_schedules_from_now_not_from_the_missed_slots(database):
    user = _user()
    await store.save_sync_policy(user, user.id, HOURLY, now=NOW)
    await store.pause_sync_policy(user, user.id, now=NOW)
    later = datetime(2026, 11, 9, 7, 30, tzinfo=timezone.utc)

    row = await store.resume_sync_policy(user, user.id, now=later)

    assert (row.paused, row.paused_by, row.paused_at) == (False, None, None)
    assert row.next_due_at == datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_resuming_a_running_policy_keeps_its_pending_slot(database):
    """A slot already due must still run: resume moves only a paused row."""
    user = _user()
    saved = await store.save_sync_policy(user, user.id, HOURLY, now=NOW)

    row = await store.resume_sync_policy(
        user, user.id, now=datetime(2026, 11, 9, tzinfo=timezone.utc)
    )

    assert row.next_due_at.replace(tzinfo=timezone.utc) == saved.next_due_at


@pytest.mark.asyncio
async def test_only_the_owner_pauses_or_resumes_and_a_missing_row_is_not_found(database):
    owner, stranger = _user(), _user()
    member_tenant = await _add_tenant(database, owner)
    await store.save_sync_policy(owner, owner.id, HOURLY, now=NOW)
    await store.save_sync_policy(owner, member_tenant, HOURLY, now=NOW)
    agent = await _add_agent(database, owner)

    for caller, principal_id in (
        (stranger, owner.id),
        (agent, owner.id),
        (stranger, member_tenant),
    ):
        for change in (store.pause_sync_policy, store.resume_sync_policy):
            with pytest.raises(SyncPolicyNotFoundError):
                await change(caller, principal_id, now=NOW)
    # A connection without its own row has nothing to pause: the default is not copied.
    with pytest.raises(SyncPolicyNotFoundError):
        await store.pause_sync_policy(owner, owner.id, uuid4(), now=NOW)
    assert await _count(database, paused=True) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["pause_sync_policy", "resume_sync_policy"])
async def test_a_change_that_loses_a_race_with_a_delete_raises_not_found(database, change):
    """Resuming a running row changes nothing; it must still not report a row that is gone."""
    user = _user()
    await store.save_sync_policy(user, user.id, HOURLY, now=NOW)
    database.before_commit = lambda: database.execute("DELETE FROM integration_sync_policies")

    with pytest.raises(SyncPolicyNotFoundError):
        await getattr(store, change)(user, user.id, now=NOW)


@pytest.mark.asyncio
async def test_a_paused_default_holds_every_connection_and_a_paused_row_only_its_own(database):
    owner = _user()
    inheriting, overriding, other = uuid4(), uuid4(), uuid4()
    await store.save_sync_policy(owner, owner.id, DAILY, now=NOW)
    await store.save_sync_policy(owner, owner.id, HOURLY, overriding, now=NOW)
    await store.save_sync_policy(owner, owner.id, HOURLY, other, now=NOW)

    async def levels():
        return [
            await store.get_sync_pause_level(credential_id=credential_id, principal_id=owner.id)
            for credential_id in (inheriting, overriding, other)
        ]

    await store.pause_sync_policy(owner, owner.id, other, now=NOW)
    assert await levels() == [None, None, "connection"]
    await store.require_sync_not_paused(credential_id=inheriting, principal_id=owner.id)
    with pytest.raises(SyncPausedError, match="connection level"):
        await store.require_sync_not_paused(credential_id=other, principal_id=owner.id)

    await store.pause_sync_policy(owner, owner.id, now=NOW)
    assert await levels() == ["default", "default", "default"]
    with pytest.raises(SyncPausedError, match="default level"):
        await store.require_sync_not_paused(credential_id=overriding, principal_id=owner.id)


@pytest.mark.asyncio
async def test_a_webhook_or_install_sync_is_dropped_only_while_a_pause_holds_the_connection(
    database,
):
    """The same rule as the scheduler and the manual route, read through the owner (SDK-1434)."""
    from cognee.modules.integrations.ingestion import sync_is_paused

    owner = _user()
    credential = SimpleNamespace(id=uuid4(), workspace_id=None, user_id=owner.id)
    assert not await sync_is_paused("github", credential, "push delivery")

    await store.save_sync_policy(owner, owner.id, DAILY, now=NOW)
    await store.pause_sync_policy(owner, owner.id, now=NOW)
    assert await sync_is_paused("github", credential, "push delivery")

    await store.resume_sync_policy(owner, owner.id, now=NOW)
    assert not await sync_is_paused("github", credential, "push delivery")


@pytest.mark.asyncio
async def test_the_caller_reads_the_effective_policy_with_its_level_only_as_owner(database):
    owner, stranger = _user(), _user()
    credential_id = uuid4()
    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_caller_effective_sync_policy(owner, owner.id, credential_id)

    await store.save_sync_policy(owner, owner.id, DAILY, now=NOW)
    inherited, inherited_held = await store.get_caller_effective_sync_policy(
        owner, owner.id, credential_id
    )
    await store.save_sync_policy(owner, owner.id, HOURLY, credential_id, now=NOW)
    own, own_held = await store.get_caller_effective_sync_policy(owner, owner.id, credential_id)

    assert store.policy_level(inherited) == "default"
    assert store.policy_level(own) == "connection"
    assert inherited_held is None and own_held is None
    with pytest.raises(SyncPolicyNotFoundError):
        await store.get_caller_effective_sync_policy(stranger, owner.id, credential_id)


def test_a_row_edited_outside_the_store_fails_validation_instead_of_hanging_next_due():
    row = SimpleNamespace(interval_minutes=0, weekdays=[0], timezone="UTC")

    with pytest.raises(ValidationError):
        store.stored_policy(row)
