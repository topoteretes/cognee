"""Shared fixtures for the integration unit tests."""

import base64
import contextlib
import importlib
import sqlite3
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy
from cognee.modules.users.models import Tenant, User
from cognee.modules.users.models.Principal import Principal

# Every module the sync-policy code reads the relational engine through.
_SYNC_POLICY_ENGINE_READERS = (
    "cognee.modules.integrations.credentials",
    "cognee.modules.integrations.scheduled_sync",
    "cognee.modules.integrations.sync_policy.store",
    "cognee.modules.integrations.sync_policy.tick",
    "cognee.modules.users.methods.get_visible_user_ids",
)


@pytest.fixture(autouse=True)
def credentials_key(monkeypatch):
    """A server configured with a credentials key, as connecting requires: the key encrypts
    tokens and every integration's OAuth state is signed with a key derived from it. Tests
    about a missing key remove it themselves."""
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEYS", raising=False)


@pytest_asyncio.fixture
async def credential_db(monkeypatch):
    """The credential store on a real in-memory SQLite table.

    For tests where the behaviour under test is the SQL itself, such as the
    compare-and-swap that keeps a stale refresh from overwriting a newer row.
    """
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        store, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    yield
    await engine.dispose()


class _Database:
    """What the code under test uses of the relational engine, over one SQLite file.

    ``before_commit`` runs once, inside the next commit of any session and before it
    writes, so a test can change the database from another connection at the moment
    a save has read a row and not yet written it. ``after_commit`` runs once, right
    after the next commit, when the save has written and not yet returned.
    """

    def __init__(self, engine, path):
        self.engine = engine
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
    """Sync policies, credentials and users on one SQLite file.

    A file, not memory, so a race test can act from a second connection between a
    read and the commit that follows, and a second process can open it.
    """
    path = tmp_path / "policies.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as connection:
        await connection.run_sync(
            IntegrationSyncPolicy.metadata.create_all,
            tables=[
                Principal.__table__,
                User.__table__,
                Tenant.__table__,
                IntegrationCredential.__table__,
                IntegrationSyncPolicy.__table__,
            ],
        )
    db = _Database(engine, path)
    for module in _SYNC_POLICY_ENGINE_READERS:
        monkeypatch.setattr(importlib.import_module(module), "get_relational_engine", lambda: db)
    yield db
    await engine.dispose()
