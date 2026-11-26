"""``reembed_dataset`` on a real dataset: the registry row, the dataset context and
a vector store per dataset, with deterministic stand-ins for the embedding models.

What is checked here is what the store-level tests cannot see: the dataset's
record, who may enter the dataset while a re-embed is unfinished, which model the
dataset's engine embeds with during and after the call, and what happens to the
vectors that are not embeddings of text (truth centroids, routing rows).
"""

from __future__ import annotations

import asyncio
import importlib
from contextvars import copy_context
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import select

import cognee.infrastructure.databases.vector.create_vector_engine as engine_module
import cognee.infrastructure.databases.vector.embeddings as embeddings_package
import cognee.infrastructure.databases.vector.embeddings.config as embedding_settings
from cognee.context_global_variables import (
    embedding_config as embedding_config_override,
)
from cognee.context_global_variables import (
    set_database_global_context_variables,
)
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
    DIMENSIONS_KEY,
    MODEL_KEY,
    REEMBED_KEY,
    STAGED_COLLECTIONS_KEY,
)
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.config import (
    EmbeddingConfig,
    get_embedding_context_config,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.exceptions import (
    DatasetReembedIncompleteError,
    EmbeddingDimensionMismatchError,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.methods import create_dataset, delete_dataset, reembed_dataset
from cognee.modules.truth_subspace.constants import TRUTH_CENTROID_COLLECTION
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase
from cognee.tasks.storage.reembed_collections import STAGING_PREFIX

context_module = importlib.import_module("cognee.context_global_variables")
routing = importlib.import_module("cognee.modules.search.dataset_routing")
reembed_collections = importlib.import_module("cognee.tasks.storage.reembed_collections")

OLD = EmbeddingConfig(embedding_provider="custom", embedding_model="old", embedding_dimensions=3)
NEW = EmbeddingConfig(embedding_provider="custom", embedding_model="new", embedding_dimensions=5)
OTHER = EmbeddingConfig(
    embedding_provider="custom", embedding_model="other", embedding_dimensions=7
)


class _Model:
    """The embedding engine of one config: a vector names its model's width on axis 0."""

    # model name -> how many texts it embeds before the provider goes down
    fails_after: dict[str, int] = {}
    embedded: list[str] = []
    # (text, entered, release): embedding this text waits for ``release``
    held: tuple[str, asyncio.Event, asyncio.Event] | None = None

    def __init__(self, config: EmbeddingConfig):
        self.model = config.embedding_model
        self.width = config.embedding_dimensions

    def get_vector_size(self) -> int:
        return self.width

    def get_batch_size(self) -> int:
        return 1

    def vector(self, text: str) -> list[float]:
        values = [float(self.width)] + [0.0] * (self.width - 1)
        values[1 + len(text) % (self.width - 1)] = 1.0
        return values

    async def embed_text(self, texts: list[str]) -> EmbeddingResult:
        limit = self.fails_after.get(self.model)
        if limit is not None and len([t for t in self.embedded if t[0] == self.model]) >= limit:
            raise ConnectionError("embedding provider is down")
        self.embedded.extend((self.model, text) for text in texts)
        if self.held is not None and texts == [self.held[0]]:
            self.held[1].set()
            await self.held[2].wait()
        return EmbeddingResult([self.vector(text) for text in texts], tokens=len(texts))


class _Chunk(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


class _Centroid(BaseModel):
    slot: int


CHUNKS = [_Chunk(text="a", belongs_to_set=["alpha"]), _Chunk(text="bb"), _Chunk(text="ccc")]


class _Routing:
    """Dataset routing, as ``reembed_dataset`` calls it.

    It records the dataset ids only. The engine a call is handed must not be kept: a
    reference held past the test would put off the engine's close until the test's event
    loop is gone, and its worker process would outlive the test.
    """

    def __init__(self):
        self.removed: list = []
        self.copied: list = []
        # What the copy does besides being recorded: an async callable, or nothing.
        self.during_copy = None

    async def remove_dataset_routes(self, dataset_id, document_ids):
        self.removed.append(dataset_id)

    async def copy_dataset_routes(self, dataset_id, index_engine):
        self.copied.append(dataset_id)
        if self.during_copy is not None:
            await self.during_copy()
        return 0


@pytest_asyncio.fixture
async def dataset(monkeypatch):
    """A dataset built with the OLD model: three chunks."""
    # A dataset has a vector store of its own only with access control on, and the
    # suite does not run with it on everywhere (a test module turns it off at import).
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")

    # An engine embeds with the config in force when it is created, as in production.
    def engine_of_the_config_in_force():
        return _Model(get_embedding_context_config())

    monkeypatch.setattr(engine_module, "get_embedding_engine", engine_of_the_config_in_force)
    monkeypatch.setattr(embeddings_package, "get_embedding_engine", engine_of_the_config_in_force)
    monkeypatch.setattr(_Model, "fails_after", {})
    monkeypatch.setattr(_Model, "embedded", [])
    monkeypatch.setattr(_Model, "held", None)
    dataset_routing = _Routing()
    monkeypatch.setattr(routing, "remove_dataset_routes", dataset_routing.remove_dataset_routes)
    monkeypatch.setattr(routing, "copy_dataset_routes", dataset_routing.copy_dataset_routes)

    user = await get_default_user()
    async with get_relational_engine().get_async_session() as session:
        created = await create_dataset(f"reembed_{uuid4().hex[:8]}", user, session)
        await session.commit()

    async with set_database_global_context_variables(created.id, embedding_config=OLD):
        await (await get_vector_engine_async()).index_data_points("_Chunk", "text", CHUNKS)

    created.routing = dataset_routing
    try:
        yield created
    finally:
        await delete_dataset(created)


async def _record(dataset_id) -> dict:
    async with get_relational_engine().get_async_session() as session:
        row = await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
        )
    return dict(row.vector_database_connection_info)


async def _chunk_vectors(store) -> dict[str, list[float]]:
    """The stored vector of every chunk, by its text."""
    rows = await store.retrieve(
        "_Chunk_text", await store.get_data_point_ids("_Chunk_text"), include_vector=True
    )
    return {row.payload["text"]: row.payload["vector"] for row in rows}


async def _assert_used_like_a_dataset_of(config: EmbeddingConfig, dataset_id) -> None:
    """Every chunk holds ``config``'s vector, and the dataset reads and writes with it."""
    model = _Model(config)
    async with set_database_global_context_variables(dataset_id, embedding_config=config):
        store = await get_vector_engine_async()
        # The engine a user of the dataset gets embeds with their model, not a leftover one.
        assert store.embedding_engine.model == config.embedding_model
        vectors = await _chunk_vectors(store)
        assert vectors == {c.text: pytest.approx(model.vector(c.text)) for c in CHUNKS}
        found = await store.search("_Chunk_text", query_text="bb", limit=1, include_payload=True)
        assert found[0].payload["text"] == "bb"
        added = _Chunk(text="dddd")
        await store.index_data_points("_Chunk", "text", [added])
        assert (await _chunk_vectors(store))["dddd"] == pytest.approx(model.vector("dddd"))
        await store.delete_data_points("_Chunk_text", [added.id])
        staged = [n for n in await store.get_collection_names() if n.startswith(STAGING_PREFIX)]
        assert staged == []


def _names(collections: list[str]) -> list[str]:
    """Collection names as LanceDB and pgvector list them; Turso lists them in lowercase."""
    return [name.casefold() for name in collections]


@pytest.mark.asyncio
async def test_a_dataset_is_rebuilt_and_recorded_with_the_new_model(dataset):
    assert (await _record(dataset.id))[MODEL_KEY] == "old"

    result = await reembed_dataset(dataset.id, NEW)

    assert (result.embedding_model, result.embedding_dimensions) == ("new", 5)
    assert _names(result.collections) == ["_chunk_text"]
    assert result.rows_embedded == len(CHUNKS)
    record = await _record(dataset.id)
    assert (record[MODEL_KEY], record[DIMENSIONS_KEY]) == ("new", 5)
    assert REEMBED_KEY not in record

    # The dataset is now used with the new model, like one built with it ...
    await _assert_used_like_a_dataset_of(NEW, dataset.id)
    # ... and refuses the old one, as any dataset refuses a model that did not build it.
    with pytest.raises(EmbeddingDimensionMismatchError):
        async with set_database_global_context_variables(dataset.id, embedding_config=OLD):
            pass

    # Its routing rows held the old vectors: removed, then copied from the rebuilt
    # store once the record says so (the copy itself decides whether the models match).
    assert dataset.routing.removed == [dataset.id]
    assert dataset.routing.copied == [dataset.id]


@pytest.mark.asyncio
async def test_without_a_config_the_dataset_moves_to_the_deployment_model(dataset, monkeypatch):
    """No ``embedding_config``: the model is the one the deployment is configured with."""
    monkeypatch.setattr(embedding_settings, "get_embedding_config", lambda: NEW)

    result = await reembed_dataset(dataset.id)

    assert (result.embedding_model, result.embedding_dimensions) == ("new", 5)
    record = await _record(dataset.id)
    assert (record[MODEL_KEY], record[DIMENSIONS_KEY]) == ("new", 5)
    # The dataset is then used as every call of the deployment uses it: with no config.
    new_model = _Model(NEW)
    async with set_database_global_context_variables(dataset.id):
        store = await get_vector_engine_async()
        assert store.embedding_engine.model == "new"
        vectors = await _chunk_vectors(store)
    assert vectors == {c.text: pytest.approx(new_model.vector(c.text)) for c in CHUNKS}


@pytest.mark.asyncio
async def test_a_failed_routing_copy_is_raised_with_the_dataset_converted(dataset):
    async def the_routing_store_is_down():
        raise ConnectionError("the routing store is down")

    dataset.routing.during_copy = the_routing_store_is_down

    with pytest.raises(ConnectionError):
        await reembed_dataset(dataset.id, NEW)

    record = await _record(dataset.id)
    assert (record[MODEL_KEY], record[DIMENSIONS_KEY]) == ("new", 5)
    assert REEMBED_KEY not in record
    await _assert_used_like_a_dataset_of(NEW, dataset.id)


@pytest.mark.asyncio
async def test_a_search_in_flight_when_the_call_ends_still_finishes(dataset, monkeypatch):
    """The call evicts the engine it used; an operation holding it keeps it until done."""
    entered, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(_Model, "held", ("a held query", entered, release))
    searches = []

    async def search():
        async with set_database_global_context_variables(dataset.id, embedding_config=NEW):
            store = await get_vector_engine_async()
            return await store.search("_Chunk_text", query_text="a held query", limit=1)

    async def a_search_starts():
        # The record is finished by now, so a user of the new model is admitted.
        searches.append(asyncio.create_task(search()))
        await entered.wait()

    dataset.routing.during_copy = a_search_starts

    await reembed_dataset(dataset.id, NEW)
    release.set()

    assert len(await searches[0]) == 1


@pytest.mark.asyncio
async def test_a_failed_embedding_leaves_the_dataset_as_it_was(dataset, monkeypatch):
    # One row per step, so the provider goes down with some rows staged.
    monkeypatch.setattr(reembed_collections, "ROWS_PER_STEP", 1)
    monkeypatch.setattr(_Model, "fails_after", {"new": 2})

    with pytest.raises(ConnectionError):
        await reembed_dataset(dataset.id, NEW)

    assert len([text for model, text in _Model.embedded if model == "new"]) == 2
    record = await _record(dataset.id)
    assert (record[MODEL_KEY], record[DIMENSIONS_KEY]) == ("old", 3)
    assert REEMBED_KEY not in record
    # Nothing staged is left, and the same process goes on with the old model.
    await _assert_used_like_a_dataset_of(OLD, dataset.id)
    assert dataset.routing.removed == []


@pytest.mark.asyncio
async def test_an_unfinished_reembed_blocks_the_dataset_until_it_is_run_again(dataset, monkeypatch):
    replace = reembed_collections.replace_collections

    async def the_process_dies(vector_engine, collection_names):
        raise OSError("died")

    monkeypatch.setattr(reembed_collections, "replace_collections", the_process_dies)
    with pytest.raises(OSError):
        await reembed_dataset(dataset.id, NEW)

    unfinished = (await _record(dataset.id))[REEMBED_KEY]
    assert unfinished[MODEL_KEY] == "new"
    assert _names(unfinished[STAGED_COLLECTIONS_KEY]) == ["_chunk_text"]
    # Neither model may use the dataset: its vectors are in two states.
    for config in (OLD, NEW):
        with pytest.raises(DatasetReembedIncompleteError):
            async with set_database_global_context_variables(dataset.id, embedding_config=config):
                pass
    # A different model cannot take over what the first one staged.
    with pytest.raises(CogneeValidationError, match="unfinished re-embed with 'new'"):
        await reembed_dataset(dataset.id, OTHER)

    monkeypatch.setattr(reembed_collections, "replace_collections", replace)
    result = await reembed_dataset(dataset.id, NEW)

    assert result.rows_embedded == 0  # everything was staged by the first call
    assert _names(result.collections) == ["_chunk_text"]
    assert REEMBED_KEY not in await _record(dataset.id)
    await _assert_used_like_a_dataset_of(NEW, dataset.id)


@pytest.mark.asyncio
async def test_a_dataset_in_use_with_another_model_is_refused_unchanged(dataset, monkeypatch):
    """Something already inside the dataset creates its engine, with the old model, first."""
    cache = engine_module.vector_engine_cache

    class _CacheWithASearchInFlight:
        """The engine cache, where the first eviction is followed by another task's use."""

        evictions = 0

        def __getattr__(self, name):
            return getattr(cache, name)

        def evict(self, force_close=False, **config):
            evicted = cache.evict(force_close=force_close, **config)
            type(self).evictions += 1
            if self.evictions == 1:

                def open_with_the_old_model():
                    embedding_config_override.set(OLD)
                    engine_module.create_vector_engine(**config)

                copy_context().run(open_with_the_old_model)
            return evicted

    monkeypatch.setattr(engine_module, "vector_engine_cache", _CacheWithASearchInFlight())

    with pytest.raises(CogneeSystemError, match="something else is using the dataset"):
        await reembed_dataset(dataset.id, NEW)

    monkeypatch.setattr(engine_module, "vector_engine_cache", cache)
    record = await _record(dataset.id)
    assert (record[MODEL_KEY], record[DIMENSIONS_KEY]) == ("old", 3)
    assert REEMBED_KEY not in record
    assert not [model for model, _ in _Model.embedded if model == "new"]
    await _assert_used_like_a_dataset_of(OLD, dataset.id)


@pytest.mark.asyncio
async def test_a_dataset_with_a_truth_subspace_is_refused_unchanged(dataset):
    """Its centroids are computed from vectors, and dropping them restarts their epoch."""
    centroid_id = uuid4()
    async with set_database_global_context_variables(dataset.id, embedding_config=OLD):
        store = await get_vector_engine_async()
        await store.upsert_raw_vectors(
            TRUTH_CENTROID_COLLECTION,
            [{"id": centroid_id, "vector": [0.0, 1.0, 0.0], "payload": {"slot": 0}}],
            payload_schema=_Centroid,
        )

    with pytest.raises(CogneeValidationError, match="has a truth subspace"):
        await reembed_dataset(dataset.id, NEW)

    assert REEMBED_KEY not in await _record(dataset.id)
    assert not _Model.embedded[len(CHUNKS) :]  # nothing was embedded for it
    async with set_database_global_context_variables(dataset.id, embedding_config=OLD):
        store = await get_vector_engine_async()
        assert len(await store.get_data_point_ids(TRUTH_CENTROID_COLLECTION)) == 1
        # Without its rows the collection is only an old-width table: dropped, not refused.
        await store.delete_data_points(TRUTH_CENTROID_COLLECTION, [centroid_id])

    await reembed_dataset(dataset.id, NEW)

    async with set_database_global_context_variables(dataset.id, embedding_config=NEW):
        assert not await (await get_vector_engine_async()).has_collection(TRUTH_CENTROID_COLLECTION)
    await _assert_used_like_a_dataset_of(NEW, dataset.id)


@pytest.mark.asyncio
async def test_a_store_that_cannot_list_its_collections_is_refused_unrecorded(dataset):
    async with set_database_global_context_variables(dataset.id, embedding_config=OLD):
        adapter_class = type((await get_vector_engine_async()).__wrapped__)

    async def not_implemented(self):
        raise NotImplementedError("get_collection_names is not implemented for this adapter")

    # Its own patch, undone here: the fixture's patches must outlive the test body.
    with pytest.MonkeyPatch.context() as adapter_patch:
        adapter_patch.setattr(adapter_class, "get_collection_names", not_implemented)
        with pytest.raises(NotImplementedError):
            await reembed_dataset(dataset.id, NEW)

    assert REEMBED_KEY not in await _record(dataset.id)
    assert not [model for model, _ in _Model.embedded if model == "new"]
    await _assert_used_like_a_dataset_of(OLD, dataset.id)


@pytest.mark.asyncio
async def test_one_dataset_cannot_be_reembedded_when_datasets_share_a_store(monkeypatch):
    monkeypatch.setattr(context_module, "backend_access_control_enabled", lambda: False)

    with pytest.raises(CogneeValidationError, match="backend access control off"):
        await reembed_dataset(uuid4(), NEW)
