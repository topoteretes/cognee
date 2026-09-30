"""Fixtures for the provenance-ledger unit tests.

A private async SQLite database (one file per test, via ``tmp_path``) stands in
for the shared relational DB: only the ``provenance_entries`` table is created,
``cognee.modules.provenance.storage.get_async_session`` is patched to hand out
sessions bound to it, and both cached factories are cleared so no state leaks
between tests.
"""

from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.relational import Base
from cognee.modules.provenance import storage as provenance_storage
from cognee.modules.provenance.manager import get_provenance_manager
from cognee.modules.provenance.models import ProvenanceEntryRow


@pytest_asyncio.fixture
async def provenance_engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/provenance.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[ProvenanceEntryRow.__table__])
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def manager(provenance_engine, monkeypatch):
    session_maker = async_sessionmaker(provenance_engine, expire_on_commit=False)

    @asynccontextmanager
    async def fake_get_async_session(auto_commit=False):
        async with session_maker() as session:
            yield session
            if auto_commit:
                await session.commit()

    monkeypatch.setattr(provenance_storage, "get_async_session", fake_get_async_session)
    get_provenance_manager.cache_clear()
    yield get_provenance_manager()
    get_provenance_manager.cache_clear()


@pytest.fixture
def tamper(provenance_engine):
    """Run raw SQL against the ledger, bypassing the manager (tamper simulation)."""

    async def _run(sql, params=None):
        async with provenance_engine.begin() as connection:
            await connection.execute(text(sql), params or {})

    return _run
