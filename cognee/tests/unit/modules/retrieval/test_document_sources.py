"""A cited connector chunk carries its document's title and link, read from its data record."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.context_global_variables import current_dataset_id
from cognee.modules.data.models import Data, Dataset
from cognee.modules.retrieval.utils import document_sources
from cognee.modules.retrieval.utils.document_sources import name_chunk_documents

DATASET_A = uuid4()
DATASET_B = uuid4()
TITLE = "Orion  launch\tplan"
URL = 'HTTPS://wiki.example.com/orion?view=a|b&q="exact"'


@pytest_asyncio.fixture
async def relational(monkeypatch):
    """Two datasets on SQLite: a connector row in each, and a plain labelled note in A.

    The connector row's label differs from its title, as after a rename in Cognee Cloud:
    the payload must carry the title, never the label.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Dataset.__table__.create)
        await conn.run_sync(Data.__table__.create)
    rows = {
        DATASET_A: Data(
            id=uuid4(),
            dataset_id=DATASET_A,
            label="Renamed in Cloud",
            system_metadata={"source": "wiki", "title": TITLE, "url": URL},
        ),
        DATASET_B: Data(
            id=uuid4(),
            dataset_id=DATASET_B,
            label="Other dataset's secret",
            system_metadata={"source": "wiki", "title": "Other dataset's secret"},
        ),
    }
    note = Data(id=uuid4(), dataset_id=DATASET_A, label="Budget call notes", system_metadata={})
    odd = Data(
        id=uuid4(), dataset_id=DATASET_A, label="Odd", system_metadata={"title": 42, "url": ["x"]}
    )
    async with sessions() as db:
        owner = uuid4()
        db.add_all([Dataset(id=d, name=str(d), owner_id=owner) for d in rows])
        db.add_all([*rows.values(), note, odd])
        await db.commit()

    @asynccontextmanager
    async def get_async_session():
        async with sessions() as session:
            yield session

    monkeypatch.setattr(
        document_sources,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=get_async_session),
    )
    yield {
        **{dataset: str(row.id) for dataset, row in rows.items()},
        "note": str(note.id),
        "odd": str(odd.id),
    }
    await engine.dispose()


def _chunk(data_id: str) -> SimpleNamespace:
    """A retrieved chunk as the vector stores return it (``ScoredResult``)."""
    return SimpleNamespace(payload={"document_id": data_id, "document_name": "text_5f1c"})


async def _name(chunks: list, dataset) -> None:
    """Name ``chunks`` inside the database context of ``dataset``, as a search does."""
    token = current_dataset_id.set(dataset)
    try:
        await name_chunk_documents(chunks)
    finally:
        current_dataset_id.reset(token)


@pytest.mark.asyncio
async def test_a_connector_chunk_carries_its_title_and_link_as_stored(relational):
    """The title from system_metadata, never the label; both exactly as the connector wrote them."""
    chunk = _chunk(relational[DATASET_A])

    await _name([chunk], DATASET_A)

    assert chunk.payload["document_title"] == TITLE
    assert chunk.payload["document_url"] == URL


@pytest.mark.asyncio
async def test_a_plain_document_payload_gains_nothing(relational):
    """A label given to ``add`` is not a connector title: the payload is untouched."""
    chunk = _chunk(relational["note"])

    await _name([chunk], DATASET_A)

    assert chunk.payload == {"document_id": relational["note"], "document_name": "text_5f1c"}


@pytest.mark.asyncio
async def test_only_the_searched_datasets_records_are_read(relational):
    """The scope is the search's own database context: another dataset's row is not read."""
    chunk = _chunk(relational[DATASET_B])

    await _name([chunk], DATASET_A)

    assert "document_title" not in chunk.payload


@pytest.mark.asyncio
async def test_without_a_dataset_the_documents_are_read_by_id(relational):
    """A search without a dataset context reads the ids this search's chunks carry."""
    chunk = _chunk(relational[DATASET_B])

    await _name([chunk], None)

    assert chunk.payload["document_title"] == "Other dataset's secret"
    assert "document_url" not in chunk.payload


@pytest.mark.asyncio
async def test_a_database_error_leaves_the_chunk_unnamed(monkeypatch):
    @asynccontextmanager
    async def failing_session():
        raise OperationalError("SELECT", {}, Exception("database is down"))
        yield

    monkeypatch.setattr(
        document_sources,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=failing_session),
    )
    chunk = _chunk(str(uuid4()))

    await _name([chunk], DATASET_A)

    assert "document_title" not in chunk.payload


@pytest.mark.asyncio
async def test_a_programming_error_is_not_swallowed(monkeypatch):
    def broken_engine():
        raise AttributeError("no such column")

    monkeypatch.setattr(document_sources, "get_relational_engine", broken_engine)

    with pytest.raises(AttributeError):
        await _name([_chunk(str(uuid4()))], DATASET_A)


@pytest.mark.asyncio
async def test_a_chunk_whose_document_id_is_not_a_uuid_is_skipped(relational):
    """``document_id`` is a free-form string on a chunk (a custom chunker, a community store);
    such a chunk is cited by its file as before, and the others are still named."""
    odd, connector = _chunk("legacy-doc-1"), _chunk(relational[DATASET_A])

    await _name([odd, connector], DATASET_A)

    assert odd.payload == {"document_id": "legacy-doc-1", "document_name": "text_5f1c"}
    assert connector.payload["document_title"] == TITLE


@pytest.mark.asyncio
async def test_a_title_or_link_that_is_not_text_is_left_out(relational):
    """A hand-built DataItem may set the keys to anything; only text is carried."""
    chunk = _chunk(relational["odd"])

    await _name([chunk], DATASET_A)

    assert "document_title" not in chunk.payload
    assert "document_url" not in chunk.payload
