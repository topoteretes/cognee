"""get_selected_dataset_data against an isolated in-memory database."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.exceptions import CogneeValidationError
from cognee.modules.data.models import Data


@pytest_asyncio.fixture
async def selection(monkeypatch):
    module = importlib.import_module("cognee.modules.data.methods.get_selected_dataset_data")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    async with engine.begin() as connection:
        await connection.run_sync(Data.__table__.create)

    dataset_id = uuid4()
    own = Data(id=uuid4(), dataset_id=dataset_id, name="current")
    legacy = Data(id=uuid4(), dataset_id=None, name="legacy")
    foreign = Data(id=uuid4(), dataset_id=uuid4(), name="foreign")
    async with sessions() as session:
        session.add_all([own, legacy, foreign])
        await session.commit()

    yield SimpleNamespace(
        select=module.get_selected_dataset_data,
        dataset_id=dataset_id,
        own=own,
        legacy=legacy,
        foreign=foreign,
    )
    await engine.dispose()


def _stale(row):
    """A detached copy of ``row`` as a caller may have loaded it earlier."""
    return Data(id=row.id, dataset_id=row.dataset_id, name="stale")


@pytest.mark.asyncio
async def test_mixed_selection_keeps_order_reads_rows_again_and_drops_deleted(selection):
    deleted = Data(id=uuid4(), dataset_id=selection.dataset_id, name="deleted")
    text = "plain item"

    result = await selection.select(
        selection.dataset_id, [_stale(selection.own), text, deleted, _stale(selection.legacy)]
    )

    assert [item if isinstance(item, str) else item.name for item in result] == [
        "current",
        text,
        "legacy",
    ]


@pytest.mark.asyncio
async def test_row_of_another_dataset_raises(selection):
    with pytest.raises(CogneeValidationError, match=str(selection.foreign.id)):
        await selection.select(selection.dataset_id, [selection.own, selection.foreign])


@pytest.mark.asyncio
async def test_single_row_and_tuple_come_back_as_lists(selection):
    single = await selection.select(selection.dataset_id, _stale(selection.own))
    pair = await selection.select(selection.dataset_id, (_stale(selection.own), "text"))

    assert [row.name for row in single] == ["current"]
    assert isinstance(pair, list)
    assert [pair[0].name, pair[1]] == ["current", "text"]


@pytest.mark.asyncio
async def test_selection_without_data_rows_is_returned_unchanged(selection):
    items = ["text", {"id": "not a row"}]

    assert await selection.select(selection.dataset_id, items) is items


@pytest.mark.asyncio
async def test_selection_of_deleted_rows_only_is_empty(selection):
    deleted = Data(id=uuid4(), dataset_id=selection.dataset_id)

    assert await selection.select(selection.dataset_id, [deleted]) == []
