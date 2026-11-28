"""A pinned data id resolves to the data item ``ingest_data`` stores it under.

``ingest_data`` stores a ``DataItem`` with a pinned ``data_id`` under
``resolve_data_id(dataset_id, pin)``, which maps a legacy id to the canonical
data item. The incremental pipeline must resolve the pin the same way, or it
looks up, marks and reports an id that has no data item behind it.
"""

import importlib
import math
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, insert

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.methods.get_existing_data_ids import DATA_ID_BATCH_SIZE
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.tasks.ingestion.data_item import DataItem

item_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_data_item")

DATASET = UUID(int=1)
OTHER_DATASET = UUID(int=2)


@pytest_asyncio.fixture
async def relational(monkeypatch, tmp_path):
    """A real SQLite relational store that counts the sessions and statements it serves."""
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'relational.db'}")
    await engine.create_database()
    served = {"sessions": 0, "statements": 0}
    open_session = engine.get_async_session

    def counting_session():
        served["sessions"] += 1
        return open_session()

    engine.get_async_session = counting_session
    event.listen(
        engine.engine.sync_engine,
        "before_cursor_execute",
        lambda *_: served.__setitem__("statements", served["statements"] + 1),
    )
    monkeypatch.setattr(item_module, "get_relational_engine", lambda: engine)
    return SimpleNamespace(engine=engine, served=served)


async def _store(engine, *rows):
    values = [
        {
            "name": "doc",
            "content_hash": str(uuid4()),
            "raw_data_location": "file:///doc.txt",
            "owner_id": uuid4(),
            "pipeline_status": {},
            "token_count": -1,
            "legacy_id": None,
        }
        | row
        for row in rows
    ]
    async with engine.get_async_session() as session:
        await session.execute(insert(Data.__table__), values)
        await session.commit()


def _pinned(*pins):
    return [DataItem(data="text", data_id=pin) for pin in pins]


@pytest.mark.asyncio
async def test_a_pinned_legacy_id_is_reported_as_its_canonical_data_item(relational):
    legacy, canonical = uuid4(), uuid4()
    dataset = SimpleNamespace(id=DATASET, name="ds")
    # A shared row split per dataset: the original kept its id in another dataset, this
    # dataset got a fork of it. The fork was already processed, so the item is skipped.
    await _store(
        relational.engine,
        {"id": legacy, "dataset_id": OTHER_DATASET},
        {
            "id": canonical,
            "legacy_id": legacy,
            "dataset_id": DATASET,
            "pipeline_status": {"p": {str(DATASET): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
        },
    )
    [item] = _pinned(legacy)
    pinned_data = await item_module.find_pinned_data([item], DATASET)
    relational.served["sessions"] = 0

    outputs = [
        out
        async for out in item_module.run_tasks_data_item_incremental(
            data_item=item,
            dataset=dataset,
            tasks=[],
            pipeline_name="p",
            pipeline_id="p",
            pipeline_run_id=uuid4(),
            ctx=SimpleNamespace(extras={}),
            user=SimpleNamespace(id=uuid4()),
            pinned_data=pinned_data,
        )
    ]

    assert outputs[-1]["data_id"] == canonical
    # The pin was read with the run's other pins, before the item ran.
    assert relational.served["sessions"] == 0


@pytest.mark.asyncio
async def test_the_exact_id_wins_over_a_data_item_forked_from_it(relational):
    pin, forked = uuid4(), uuid4()
    await _store(
        relational.engine,
        {"id": forked, "legacy_id": pin, "dataset_id": DATASET},
        {"id": pin, "dataset_id": DATASET},
    )

    pinned_data = await item_module.find_pinned_data(_pinned(pin), DATASET)

    assert pinned_data[pin].id == pin


@pytest.mark.asyncio
async def test_an_id_unknown_in_the_dataset_is_looked_up_alone(relational):
    elsewhere, missing = uuid4(), uuid4()
    await _store(
        relational.engine,
        {"id": elsewhere, "dataset_id": OTHER_DATASET},
        # A legacy id names its fork in the fork's own dataset only.
        {"id": uuid4(), "legacy_id": missing, "dataset_id": OTHER_DATASET},
    )

    pinned_data = await item_module.find_pinned_data(_pinned(elsewhere, missing), DATASET)

    assert pinned_data[elsewhere].id == elsewhere
    # A pin with no data item is left out: the item runs and ingest_data mints it.
    assert missing not in pinned_data


@pytest.mark.asyncio
async def test_pins_with_no_row_of_their_own_take_two_queries_per_batch(relational):
    count = DATA_ID_BATCH_SIZE + 1
    # Forks in the dataset and rows of another one: no pin has a row of its own in the
    # dataset, so every pin takes both lookups.
    forks = [{"id": uuid4(), "legacy_id": uuid4(), "dataset_id": DATASET} for _ in range(count)]
    elsewhere = [{"id": uuid4(), "dataset_id": OTHER_DATASET} for _ in range(count)]
    await _store(relational.engine, *forks, *elsewhere)
    pins = [fork["legacy_id"] for fork in forks] + [row["id"] for row in elsewhere]
    relational.served.update(sessions=0, statements=0)

    pinned_data = await item_module.find_pinned_data(_pinned(*pins), DATASET)

    assert {pin: row.id for pin, row in pinned_data.items()} == {
        **{fork["legacy_id"]: fork["id"] for fork in forks},
        **{row["id"]: row["id"] for row in elsewhere},
    }
    batches = math.ceil(len(pins) / DATA_ID_BATCH_SIZE)
    assert relational.served == {"sessions": 1, "statements": 2 * batches}


@pytest.mark.asyncio
async def test_pins_with_a_row_in_the_dataset_take_one_query_per_batch(relational):
    """An unchanged sync: every pin is its own row in the dataset, so no fork is looked up."""
    pins = [uuid4() for _ in range(DATA_ID_BATCH_SIZE + 1)]
    await _store(relational.engine, *({"id": pin, "dataset_id": DATASET} for pin in pins))
    relational.served.update(sessions=0, statements=0)

    pinned_data = await item_module.find_pinned_data(_pinned(*pins), DATASET)

    assert sorted(pinned_data) == sorted(pins)
    batches = math.ceil(len(pins) / DATA_ID_BATCH_SIZE)
    assert relational.served == {"sessions": 1, "statements": batches}


@pytest.mark.asyncio
async def test_a_run_without_pins_opens_no_session(relational):
    """Content items and Data rows have no pin; their run reads nothing up front."""
    assert await item_module.find_pinned_data(["text", DataItem(data="text")], DATASET) == {}
    assert relational.served["sessions"] == 0
