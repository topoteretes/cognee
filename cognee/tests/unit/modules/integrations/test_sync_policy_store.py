"""Sync policies are scoped to the caller and inherit the owner's default (SDK-1434).

Runs on a real SQLite file with the real ``get_visible_user_ids``, as the
principal-configuration scope tests do (SDK-803). A file, not memory, so the race
tests can act from a second connection between a read and the commit that follows.
"""

import contextlib
import importlib
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Registers integration_credentials, the table the policy table's foreign key points at.
import cognee.modules.integrations.models
from cognee.modules.integrations.sync_policy import SyncPolicy, SyncPolicyNotFoundError
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy
from cognee.modules.users.models import Tenant, User
from cognee.modules.users.models.Principal import Principal

store = importlib.import_module("cognee.modules.integrations.sync_policy.store")
visible_user_ids = importlib.import_module("cognee.modules.users.methods.get_visible_user_ids")

NOW = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
DAILY = SyncPolicy(interval_minutes=1440, weekdays=[0, 1, 2, 3, 4], timezone="Europe/Belgrade")
HOURLY = SyncPolicy(interval_minutes=60)


class _Database:
    """What the code under test uses of the relational engine, over one SQLite file.

    ``before_commit`` runs once, inside the next commit of any session and before it
    writes, so a test can change the database from another connection at the moment
    a save has read a row and not yet written it. ``after_commit`` runs once, right
    after the next commit, when the save has written and not yet returned.
    """

    def __init__(self, engine, path):
        self.path = path
        self.before_commit = None
        self.after_commit = None
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        session = self._sessions()
        event.listen(session.sync_session, "before_commit", self._run_before_commit)
        event.listen(session.sync_session, "after_commit", self._run_after_commit)
        return session

    def _run_before_commit(self, _session):
        hook, self.before_commit = self.before_commit, None
        if hook:
            hook()

    def _run_after_commit(self, _session):
        hook, self.after_commit = self.after_commit, None
        if hook:
            hook()

    def execute(self, statement, parameters=()):
        with contextlib.closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(statement, parameters)


@pytest_asyncio.fixture
async def database(monkeypatch, tmp_path):
    path = tmp_path / "policies.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as connection:
        await connection.run_sync(
            IntegrationSyncPolicy.metadata.create_all,
            tables=[
                Principal.__table__,
                User.__table__,
                Tenant.__table__,
                IntegrationSyncPolicy.__table__,
            ],
        )
    db = _Database(engine, path)
    monkeypatch.setattr(store, "get_relational_engine", lambda: db)
    monkeypatch.setattr(visible_user_ids, "get_relational_engine", lambda: db)
    yield db
    await engine.dispose()


def _user():
    return SimpleNamespace(id=uuid4())


async def _add_agent(database, parent):
    """A real agent user: its ``parent_user_id`` is what get_visible_user_ids reads."""
    agent = _user()
    async with database.get_async_session() as session:
        session.add(
            User(
                id=agent.id,
                email=f"{agent.id}@example.com",
                hashed_password="x",
                is_active=True,
                is_verified=True,
                is_superuser=False,
                parent_user_id=parent.id,
            )
        )
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

    assert await store.get_effective_sync_policy(inheriting_credential, tenant_id) is None

    await store.save_sync_policy(owner, tenant_id, DAILY, now=NOW)
    await store.save_sync_policy(owner, tenant_id, HOURLY, own_credential, now=NOW)

    own = await store.get_effective_sync_policy(own_credential, tenant_id)
    inherited = await store.get_effective_sync_policy(inheriting_credential, tenant_id)
    assert own.interval_minutes == 60
    assert inherited.interval_minutes == 1440
    assert inherited.credential_id is None


@pytest.mark.asyncio
async def test_deleting_a_connection_row_returns_it_to_the_default(database):
    user, credential_id = _user(), uuid4()
    await store.save_sync_policy(user, user.id, DAILY, now=NOW)
    await store.save_sync_policy(user, user.id, HOURLY, credential_id, now=NOW)

    await store.delete_sync_policy(user, user.id, credential_id)

    assert (await store.get_effective_sync_policy(credential_id, user.id)).credential_id is None
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
async def test_a_first_save_that_loses_a_race_raises_and_does_not_overwrite(database):
    user = _user()
    # After the save has found no row, before it inserts: another connection inserts one.
    database.before_commit = lambda: _insert_default_row(database, user.id)

    with pytest.raises(IntegrityError):
        await store.save_sync_policy(user, user.id, DAILY, now=NOW)

    # The competing row is the one that stays, untouched.
    assert await _count(database, principal_id=user.id, interval_minutes=60) == 1
    assert await _count(database) == 1
