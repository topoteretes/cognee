"""Selected-row queries execute against SQLite; only the engine location is replaced."""

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected, expected, limit, offset",
    [
        (None, [1, 2, 3], None, 0),
        ([2], [2], None, 0),
        ([2, 3], [2, 3], None, 0),
        ([2, 4], [2], None, 0),
        ([4], [], None, 0),
        ([5], [], None, 0),
        ([], [], None, 0),
        ([2, 3, 4], [3], 1, 1),
    ],
    ids=[
        "unfiltered",
        "one_selected",
        "several_selected",
        "foreign_id_among_selected",
        "foreign_id_only",
        "unknown_id",
        "empty_selection",
        "filter_before_pagination",
    ],
)
async def test_data_ids_select_only_rows_in_the_dataset(
    monkeypatch, selected, expected, limit, offset
):
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    dataset_id, other_dataset_id = uuid4(), uuid4()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Data.__table__.create)
        async with sessions() as session:
            session.add_all(
                [
                    Data(id=UUID(int=1), dataset_id=dataset_id, data_size=100),
                    Data(id=UUID(int=2), dataset_id=dataset_id, data_size=20),
                    Data(id=UUID(int=3), dataset_id=dataset_id, data_size=10),
                    Data(id=UUID(int=4), dataset_id=other_dataset_id, data_size=200),
                ]
            )
            await session.commit()

        rows = await module.get_dataset_data(
            dataset_id,
            data_ids=None if selected is None else [UUID(int=i) for i in selected],
            limit=limit,
            offset=offset,
        )

        assert [row.id for row in rows] == [UUID(int=i) for i in expected]
        assert all(row.dataset_id == dataset_id for row in rows)
    finally:
        await engine.dispose()
