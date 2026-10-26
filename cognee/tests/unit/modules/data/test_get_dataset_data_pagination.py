"""Exercise the actual dataset queries against an isolated in-memory database."""

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data


@pytest.mark.asyncio
async def test_pages_are_stable_for_equal_sizes_and_count_is_dataset_scoped(monkeypatch):
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    dataset_id = uuid4()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Data.__table__.create)
        async with sessions() as session:
            # Insert ties in reverse UUID order so insertion order cannot hide a missing sort.
            session.add_all(
                [
                    Data(id=UUID(int=(15 << 124) + i), dataset_id=dataset_id, data_size=10)
                    for i in range(250, 0, -1)
                ]
            )
            session.add(Data(id=UUID(int=(15 << 124) + 251), dataset_id=dataset_id, data_size=20))
            session.add(Data(dataset_id=uuid4(), data_size=100))
            await session.commit()

        expected = [UUID(int=(15 << 124) + 251)] + [
            UUID(int=(15 << 124) + i) for i in range(1, 251)
        ]
        pages = [
            await module.get_dataset_data(dataset_id, limit=100, offset=offset)
            for offset in (0, 100, 200)
        ]
        assert [len(page) for page in pages] == [100, 100, 51]
        assert [row.id for page in pages for row in page] == expected
        assert [row.id for row in await module.get_dataset_data(dataset_id)] == expected
        assert await module.get_dataset_data(dataset_id, limit=100, offset=251) == []
        assert await module.count_dataset_data(dataset_id) == 251
        assert await module.count_dataset_data(uuid4()) == 0
    finally:
        await engine.dispose()
