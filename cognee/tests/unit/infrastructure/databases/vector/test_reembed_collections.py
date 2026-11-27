"""Re-embedding a vector store's collections, on real LanceDB and Turso files.

``stage_collections`` and ``replace_collections`` are driven the way
``reembed_dataset`` drives them: the store is first written with one embedding
model, then opened with an engine that embeds with another, of another width.
The embedding engines are deterministic stand-ins, so a stored vector tells which
model computed it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from pydantic import BaseModel

from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.databases.vector.exceptions import CollectionNotReembeddableError
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage.reembed_collections import (
    STAGING_PREFIX,
    drop_staging_collections,
    get_reembeddable_collection_names,
    replace_collections,
    stage_collections,
    staging_collection_name,
)

OLD_WIDTH = 3
NEW_WIDTH = 5
# Texts per provider request: smaller than the collection, so a write batch is
# embedded in several requests.
REQUEST_SIZE = 2


class _Model:
    """An embedding model: axis 0 says which model, the rest say which text."""

    def __init__(
        self,
        marker: float,
        width: int,
        fail_after: int | None = None,
        request_limit: int | None = None,
    ):
        self.marker = marker
        self.width = width
        self.model = f"model-{marker}"
        self.embedded: list[str] = []
        self.fail_after = fail_after
        # The most texts the provider takes in one request, when it has a limit.
        self.request_limit = request_limit

    def get_vector_size(self) -> int:
        return self.width

    def get_batch_size(self) -> int:
        return REQUEST_SIZE

    def vector(self, text: str) -> list[float]:
        values = [self.marker] + [0.0] * (self.width - 1)
        values[1 + len(text) % (self.width - 1)] = 1.0
        return values

    async def embed_text(self, texts: list[str]) -> EmbeddingResult:
        assert self.request_limit is None or len(texts) <= self.request_limit
        if self.fail_after is not None and len(self.embedded) >= self.fail_after:
            raise ConnectionError("embedding provider is down")
        self.embedded.extend(texts)
        return EmbeddingResult([self.vector(text) for text in texts], tokens=len(texts))


class _Chunk(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _Entity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _lancedb(path, model):
    pytest.importorskip("lancedb")
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    return LanceDBAdapter(
        url=str(path / "lance"), api_key=None, embedding_engine=EmbeddingGateway(model)
    )


def _turso(path, model):
    pytest.importorskip("turso")
    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
        TursoVectorAdapter,
    )

    return TursoVectorAdapter(
        url=str(path / "turso.db"), api_key=None, embedding_engine=EmbeddingGateway(model)
    )


@pytest_asyncio.fixture(params=[_lancedb, _turso], ids=["lancedb", "turso"])
async def open_store(request, tmp_path):
    """Open the one store of a test with a given embedding model, as a new engine."""
    opened = []

    def _open(model):
        adapter = request.param(tmp_path, model)
        opened.append(adapter)
        return adapter

    yield _open
    for adapter in opened:
        await adapter.close()


CHUNKS = [
    _Chunk(text="a", belongs_to_set=["alpha"]),
    _Chunk(text="bb", belongs_to_set=["alpha", "beta"]),
    _Chunk(text="ccc"),
    _Chunk(text="dddd"),
    _Chunk(text="eeeee", belongs_to_set=["beta"]),
]
ENTITIES = [_Entity(name="Acme Corp"), _Entity(name="Berlin")]


async def _write_old_store(open_store) -> _Model:
    """A store as the old model built it: two collections with rows, one empty."""
    old = _Model(marker=1.0, width=OLD_WIDTH)
    store = open_store(old)
    await store.index_data_points("_Chunk", "text", CHUNKS)
    await store.index_data_points("_Entity", "name", ENTITIES)
    await store.create_vector_index("_Summary", "text")
    return old


async def _rows(store, collection_name: str) -> dict[str, dict]:
    """Every row of a collection by id: its payload, with the vector under "vector"."""
    row_ids = await store.get_data_point_ids(collection_name)
    rows = await store.retrieve(collection_name, row_ids, include_vector=True)
    return {str(row.id): row.payload for row in rows}


def _collection(names: list[str], wanted: str) -> str:
    """A collection's name as the store lists it (Turso lists it in lowercase)."""
    return next(name for name in names if name.casefold() == wanted.casefold())


@pytest.mark.asyncio
async def test_every_collection_is_rebuilt_with_the_new_model(open_store):
    await _write_old_store(open_store)
    new = _Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE)
    store = open_store(new)
    names = await get_reembeddable_collection_names(store)
    assert [name.casefold() for name in names] == ["_chunk_text", "_entity_name", "_summary_text"]
    chunks, entities = _collection(names, "_Chunk_text"), _collection(names, "_Entity_name")
    before = await _rows(store, chunks)

    embedded = await stage_collections(store, names)
    await replace_collections(store, names)

    assert embedded == len(CHUNKS) + len(ENTITIES)
    assert sorted(new.embedded) == sorted(
        [chunk.text for chunk in CHUNKS] + [entity.name for entity in ENTITIES]
    )
    # Same collections, nothing staged left behind.
    assert await get_reembeddable_collection_names(store) == names
    assert not [n for n in await store.get_collection_names() if n.startswith(STAGING_PREFIX)]

    after = await _rows(store, chunks)
    assert set(after) == {str(chunk.id) for chunk in CHUNKS}
    for row_id, payload in after.items():
        # The vector is the new model's embedding of the row's own text ...
        assert payload["vector"] == pytest.approx(new.vector(payload["text"]))
        # ... and nothing else about the row changed.
        assert {k: v for k, v in payload.items() if k != "vector"} == {
            k: v for k, v in before[row_id].items() if k != "vector"
        }
    assert {payload["text"] for payload in (await _rows(store, entities)).values()} == {
        "Acme Corp",
        "Berlin",
    }

    # The store answers a search with the new model's width.
    found = await store.search(chunks, query_text="ccc", limit=1, include_payload=True)
    assert new.vector(found[0].payload["text"]) == new.vector("ccc")
    # A new row lands next to the rebuilt ones, and in the collection that was empty.
    await store.index_data_points("_Chunk", "text", [_Chunk(text="ffffff")])
    await store.index_data_points("_Summary", "text", [_Chunk(text="a summary")])
    assert len(await store.get_data_point_ids(chunks)) == len(CHUNKS) + 1
    assert len(await store.get_data_point_ids(_collection(names, "_Summary_text"))) == 1


@pytest.mark.asyncio
async def test_a_failed_embedding_changes_no_collection_and_the_next_call_continues(
    open_store, monkeypatch
):
    # Rows are staged a step at a time; two rows per step here, so the provider
    # fails with some of a collection staged and some not.
    monkeypatch.setattr("cognee.tasks.storage.reembed_collections.ROWS_PER_STEP", 2)
    old = await _write_old_store(open_store)
    failing = _Model(marker=2.0, width=NEW_WIDTH, fail_after=4, request_limit=REQUEST_SIZE)
    store = open_store(failing)
    names = await get_reembeddable_collection_names(store)
    chunks = _collection(names, "_Chunk_text")

    with pytest.raises(ConnectionError):
        await stage_collections(store, names)

    # Every collection still holds the old model's vectors: the old engine reads them.
    old_store = open_store(old)
    for payload in (await _rows(old_store, chunks)).values():
        assert payload["vector"] == pytest.approx(old.vector(payload["text"]))
    assert len(await old_store.search(chunks, query_text="a", limit=1)) == 1

    # The next call embeds only what the failed one did not stage.
    new = _Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE)
    store = open_store(new)
    already_staged = len(await store.get_data_point_ids(staging_collection_name(chunks)))
    assert 0 < already_staged < len(CHUNKS)

    embedded = await stage_collections(store, names)
    await replace_collections(store, names)

    assert embedded == len(CHUNKS) + len(ENTITIES) - already_staged
    assert len(new.embedded) == embedded
    assert set(await _rows(store, chunks)) == {str(chunk.id) for chunk in CHUNKS}
    for payload in (await _rows(store, chunks)).values():
        assert payload["vector"] == pytest.approx(new.vector(payload["text"]))


@pytest.mark.asyncio
async def test_a_replace_that_stopped_is_finished_by_the_next_call(open_store, monkeypatch):
    await _write_old_store(open_store)
    new = _Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE)
    store = open_store(new)
    names = await get_reembeddable_collection_names(store)
    chunks, entities = _collection(names, "_Chunk_text"), _collection(names, "_Entity_name")
    await stage_collections(store, names)

    # Stop inside the first collection: it is dropped and only partly written again.
    real_upsert = type(store).upsert_raw_vectors
    calls = []

    async def upsert_once_then_die(self, collection_name, points, payload_schema=None):
        calls.append(collection_name)
        if len(calls) > 1:
            raise OSError("the process died")
        await real_upsert(self, collection_name, points[:2], payload_schema)

    monkeypatch.setattr(type(store), "upsert_raw_vectors", upsert_once_then_die)
    monkeypatch.setattr("cognee.tasks.storage.reembed_collections.ROWS_PER_STEP", 3)
    with pytest.raises(OSError):
        await replace_collections(store, names)
    assert len(await store.get_data_point_ids(chunks)) == 2
    monkeypatch.setattr(type(store), "upsert_raw_vectors", real_upsert)

    await replace_collections(store, names)

    assert set(await _rows(store, chunks)) == {str(chunk.id) for chunk in CHUNKS}
    assert len(await _rows(store, entities)) == len(ENTITIES)
    assert not [n for n in await store.get_collection_names() if n.startswith(STAGING_PREFIX)]
    assert new.embedded and len(new.embedded) == len(CHUNKS) + len(ENTITIES)  # embedded once


def _staged(names: list[str]) -> list[str]:
    return [name for name in names if name.startswith(STAGING_PREFIX)]


@pytest.mark.asyncio
async def test_a_row_without_text_is_refused_and_staging_can_be_undone(open_store):
    await _write_old_store(open_store)
    old_store = open_store(_Model(marker=1.0, width=OLD_WIDTH))

    class _Centroid(BaseModel):
        slot: int

    # Named to be staged last: the collections before it are staged when it is refused.
    centroid_id = uuid4()
    await old_store.upsert_raw_vectors(
        "_Zentroid_vector",
        [{"id": centroid_id, "vector": [0.0, 1.0, 0.0], "payload": {"slot": 0}}],
        payload_schema=_Centroid,
    )
    store = open_store(_Model(marker=2.0, width=NEW_WIDTH))
    names = await get_reembeddable_collection_names(store)
    assert names[-1].casefold() == "_zentroid_vector"

    with pytest.raises(CollectionNotReembeddableError, match="has no text") as refused:
        await stage_collections(store, names)

    assert refused.value.collection_name.casefold() == "_zentroid_vector"
    assert UUID(str(refused.value.data_point_id)) == centroid_id
    assert len(_staged(await store.get_collection_names())) == 3  # the collections before it
    await drop_staging_collections(store)
    assert sorted(await store.get_collection_names()) == names
    assert len(await store.get_data_point_ids(_collection(names, "_Chunk_text"))) == len(CHUNKS)


@pytest.mark.asyncio
async def test_a_row_is_embedded_from_the_text_an_index_write_embeds(open_store):
    """An index write embeds the stripped text, whatever the store keeps of the padding."""
    old = _Model(marker=1.0, width=OLD_WIDTH)
    padded = _Chunk(text="  padded chunk \n")
    await open_store(old).index_data_points("_Chunk", "text", [padded])
    assert old.embedded == ["padded chunk"]
    new = _Model(marker=2.0, width=NEW_WIDTH)
    store = open_store(new)
    names = await get_reembeddable_collection_names(store)
    stored_text = (await _rows(store, names[0]))[str(padded.id)]["text"]

    await stage_collections(store, names)
    await replace_collections(store, names)

    assert new.embedded == ["padded chunk"]
    assert (await _rows(store, names[0]))[str(padded.id)]["text"] == stored_text


@pytest.mark.asyncio
async def test_staging_logs_how_far_each_collection_is(open_store, monkeypatch, caplog):
    await _write_old_store(open_store)
    store = open_store(_Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE))
    names = await get_reembeddable_collection_names(store)
    chunks = _collection(names, "_Chunk_text")
    monkeypatch.setattr("cognee.tasks.storage.reembed_collections.ROWS_PER_STEP", 2)

    with caplog.at_level("INFO"):
        await stage_collections(store, [chunks])

    said = [r.getMessage() for r in caplog.records if chunks in r.getMessage()]
    expected = [
        f"Re-embedding 5 row(s) of {chunks}",
        f"Re-embedded 2 of 5 row(s) of {chunks}",
        f"Re-embedded 4 of 5 row(s) of {chunks}",
        f"Re-embedded 5 of 5 row(s) of {chunks}",
    ]
    assert len(said) == len(expected)
    assert all(line in message for line, message in zip(expected, said, strict=True))


@pytest.mark.asyncio
async def test_staging_again_follows_what_changed_in_the_collection(open_store):
    """A second stage call (an interrupted one, continued) leaves staging equal to the rows."""
    old = await _write_old_store(open_store)
    new = _Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE)
    store = open_store(new)
    names = await get_reembeddable_collection_names(store)
    chunks = _collection(names, "_Chunk_text")
    await stage_collections(store, names)
    old_store = open_store(old)
    added = _Chunk(text="added between the calls")
    await old_store.index_data_points("_Chunk", "text", [added])
    await old_store.delete_data_points(chunks, [CHUNKS[0].id])

    assert await stage_collections(store, names) == 1  # only the added row is embedded
    await replace_collections(store, names)

    assert set(await _rows(store, chunks)) == {str(c.id) for c in [*CHUNKS[1:], added]}


@pytest.mark.asyncio
async def test_a_row_written_after_staging_is_embedded_before_its_collection_is_replaced(
    open_store,
):
    await _write_old_store(open_store)
    new = _Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE)
    store = open_store(new)
    names = await get_reembeddable_collection_names(store)
    chunks = _collection(names, "_Chunk_text")
    await stage_collections(store, names)
    late = _Chunk(text="written late")
    await open_store(_Model(marker=1.0, width=OLD_WIDTH)).index_data_points(
        "_Chunk", "text", [late]
    )

    assert await replace_collections(store, names) == 1  # the late row is embedded now

    rows = await _rows(store, chunks)
    assert len(rows) == len(CHUNKS) + 1
    assert rows[str(late.id)]["vector"] == pytest.approx(new.vector("written late"))


@pytest.mark.asyncio
async def test_a_collection_is_not_dropped_while_its_staging_lacks_a_row(open_store, monkeypatch):
    """Should a row still be unstaged when its collection is about to go, nothing is dropped."""
    await _write_old_store(open_store)
    store = open_store(_Model(marker=2.0, width=NEW_WIDTH, request_limit=REQUEST_SIZE))
    names = await get_reembeddable_collection_names(store)
    chunks = _collection(names, "_Chunk_text")
    await stage_collections(store, names)
    late = _Chunk(text="written late")
    await open_store(_Model(marker=1.0, width=OLD_WIDTH)).index_data_points(
        "_Chunk", "text", [late]
    )
    # The row arrives after the last look for unstaged rows.
    monkeypatch.setattr(
        "cognee.tasks.storage.reembed_collections._stage_unstaged_rows", AsyncMock(return_value=0)
    )

    with pytest.raises(CogneeSystemError, match="something is writing to it"):
        await replace_collections(store, names)

    assert str(late.id) in await store.get_data_point_ids(chunks)
    assert len(await store.get_data_point_ids(chunks)) == len(CHUNKS) + 1


def test_staging_names_fit_any_store():
    long_name = "A" * 80 + "_text"
    assert len(staging_collection_name(long_name)) < 63
    assert staging_collection_name(long_name) != staging_collection_name("B" + long_name[1:])
    assert staging_collection_name("DocumentChunk_text").startswith(STAGING_PREFIX)
