"""Shared fixtures for the integration unit tests."""

import base64
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


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
