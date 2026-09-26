"""A stored configuration is readable by its owner only (SDK-803).

``get_principal_configuration`` used to select by config id alone, so any
authenticated user holding another principal's config id could read it. The
lookup is now scoped to the caller: someone else's config reads exactly like a
missing one.
"""

import importlib
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration

# The ``methods`` package re-exports each function under its module's own name,
# so hold the modules themselves for patching.
get_module = importlib.import_module("cognee.modules.users.methods.get_principal_configuration")
store_module = importlib.import_module("cognee.modules.users.methods.store_principal_configuration")


class _SqliteEngine:
    """The one relational-engine method these functions use, on in-memory SQLite."""

    def __init__(self, engine):
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self._sessionmaker()


@pytest.fixture
async def relational_engine(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(
            PrincipalConfiguration.metadata.create_all, tables=[PrincipalConfiguration.__table__]
        )

    fake_engine = _SqliteEngine(engine)
    monkeypatch.setattr(get_module, "get_relational_engine", lambda: fake_engine)
    monkeypatch.setattr(store_module, "get_relational_engine", lambda: fake_engine)
    yield fake_engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_owner_reads_own_configuration(relational_engine):
    owner_id = uuid4()
    record = await store_module.store_principal_configuration(
        principal_id=owner_id, name="llm", configuration={"model": "m"}
    )

    result = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=owner_id
    )

    assert result == {"model": "m"}


@pytest.mark.asyncio
async def test_other_principal_gets_same_result_as_missing_id(relational_engine):
    owner_id = uuid4()
    record = await store_module.store_principal_configuration(
        principal_id=owner_id, name="llm", configuration={"model": "m"}
    )

    other_principal_read = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=uuid4()
    )
    missing_id_read = await get_module.get_principal_configuration(
        config_id=uuid4(), principal_id=owner_id
    )

    assert other_principal_read == {}
    assert other_principal_read == missing_id_read
