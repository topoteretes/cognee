"""The migration that moves datasets from the previous default local embedding model
to the current one, run the way the migration runner runs it.

A real dataset: its registry row, its context and a vector store of its own. The
embedding models are deterministic stand-ins, so nothing is downloaded.

Going through the runner is the point. It is the runner that has to get into a
dataset the configured model cannot serve, and it holds the dataset's queue slot
while the migration runs.
"""

from __future__ import annotations

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee.infrastructure.databases.vector.create_vector_engine as engine_module
import cognee.infrastructure.databases.vector.embeddings as embeddings_package
import cognee.infrastructure.databases.vector.embeddings.config as embedding_settings
from cognee.context_global_variables import (
    DatabaseContextManager,
    set_database_global_context_variables,
)
from cognee.context_global_variables import (
    embedding_config as embedding_config_override,
)
from cognee.exceptions import CogneeConfigurationError, CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
    DIMENSIONS_KEY,
    MODEL_KEY,
    REEMBED_KEY,
    STAGED_COLLECTIONS_KEY,
    record_reembed_progress,
)
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache
from cognee.infrastructure.databases.vector.embeddings.config import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EmbeddingConfig,
    get_embedding_context_config,
    resolve_embedding_defaults,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.exceptions import EmbeddingDimensionMismatchError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.modules import preflight
from cognee.modules.data.methods import create_dataset, delete_dataset
from cognee.modules.migrations import runner
from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions import reembed_local_default_embedding_model as migration
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase
from cognee.tasks.storage.reembed_collections import (
    STAGING_PREFIX,
    stage_collections,
    staging_collection_name,
)

routing = importlib.import_module("cognee.modules.search.dataset_routing")
reembed_collections = importlib.import_module("cognee.tasks.storage.reembed_collections")

SLUG = "reembed_local_default_embedding_model"
BEFORE = "postgres_graph_source_ref_tables"

PREVIOUS_DEFAULT = {MODEL_KEY: "BAAI/bge-small-en-v1.5", DIMENSIONS_KEY: 384}
LOCAL_DEFAULT = {MODEL_KEY: DEFAULT_LOCAL_EMBEDDING_MODEL, DIMENSIONS_KEY: 1024}

# Configured models: the previous default, as its own setting names it, and another one.
PREVIOUS = EmbeddingConfig(
    embedding_provider="fastembed",
    embedding_model=PREVIOUS_DEFAULT[MODEL_KEY],
    embedding_dimensions=PREVIOUS_DEFAULT[DIMENSIONS_KEY],
)
OTHER = EmbeddingConfig(
    embedding_provider="custom", embedding_model="other", embedding_dimensions=7
)
# What an installation with no embedding setting reads as.
NOTHING_CONFIGURED = EmbeddingConfig(
    embedding_provider="openai",
    embedding_model="openai/text-embedding-3-large",
    embedding_api_key=None,
    embedding_endpoint=None,
)


class _Model:
    """An embedding engine: a vector names its model's width on axis 0."""

    embedded: list[str] = []  # the model of every text embedded
    down: set[str] = set()  # models whose provider is down

    def __init__(self, model: str, width: int):
        self.model = model
        self.width = width

    def get_vector_size(self) -> int:
        return self.width

    def get_batch_size(self) -> int:
        return 100

    def vector(self, text: str) -> list[float]:
        values = [float(self.width)] + [0.0] * (self.width - 1)
        values[1 + len(text) % (self.width - 1)] = 1.0
        return values

    async def embed_text(self, texts: list[str]) -> EmbeddingResult:
        if self.model in self.down:
            raise ConnectionError("embedding provider is down")
        self.embedded.extend(self.model for _ in texts)
        return EmbeddingResult([self.vector(text) for text in texts], tokens=len(texts))


class _Chunk(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}


CHUNKS = [_Chunk(text="a"), _Chunk(text="bb"), _Chunk(text="ccc")]


class _Routing:
    """Dataset routing as the migration and the re-embed call it; it records the calls."""

    def __init__(self):
        self.width_matches = False
        self.calls: list[str] = []

    async def get_dataset_route_engine(self):
        return None

    async def routing_width_matches(self, engine) -> bool:
        return self.width_matches

    async def prune_dataset_routes(self) -> None:
        self.calls.append("prune")
        self.width_matches = True

    async def remove_dataset_routes(self, dataset_id, document_ids) -> None:
        self.calls.append("remove")

    async def copy_dataset_routes(self, dataset_id, index_engine) -> int:
        self.calls.append("copy")
        return 0


@pytest.fixture
def installation(monkeypatch):
    """Set what the installation has configured: ``installation(config, llm_key=...)``."""
    # A dataset has a vector store of its own only with access control on.
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")

    def engine_in_force():
        # As get_embedding_engine resolves it, local default included.
        _provider, model, dimensions = resolve_embedding_defaults(
            get_embedding_context_config(), get_llm_context_config()
        )
        return _Model(model, dimensions)

    monkeypatch.setattr(engine_module, "get_embedding_engine", engine_in_force)
    monkeypatch.setattr(embeddings_package, "get_embedding_engine", engine_in_force)
    monkeypatch.setattr(_Model, "embedded", [])
    monkeypatch.setattr(_Model, "down", set())

    def configure(config: EmbeddingConfig, llm_key: bool = False) -> None:
        monkeypatch.setattr(embedding_settings, "get_embedding_config", lambda: config)
        monkeypatch.setattr(
            preflight, "keyless_local_defaults_apply", lambda llm_config=None: not llm_key
        )

    configure(NOTHING_CONFIGURED)
    return configure


@pytest_asyncio.fixture
async def dataset(monkeypatch, installation):
    """A dataset the previous default built: three chunks, stamped before the migration."""
    dataset_routing = _Routing()
    for name in (
        "get_dataset_route_engine",
        "routing_width_matches",
        "prune_dataset_routes",
        "remove_dataset_routes",
        "copy_dataset_routes",
    ):
        monkeypatch.setattr(routing, name, getattr(dataset_routing, name))

    user = await get_default_user()
    async with get_relational_engine().get_async_session() as session:
        created = await create_dataset(f"local_default_{uuid4().hex[:8]}", user, session)
        await session.commit()
    created.routing = dataset_routing
    try:
        await _build(created.id, PREVIOUS)
        yield created
    finally:
        await delete_dataset(created)


async def _build(dataset_id, config: EmbeddingConfig) -> None:
    """Index the chunks with ``config``'s model and stamp the dataset before the migration."""
    async with set_database_global_context_variables(dataset_id, embedding_config=config):
        await (await get_vector_engine_async()).index_data_points("_Chunk", "text", CHUNKS)
    # A new dataset is stamped at head; this one is from the release before.
    await runner._stamp_dataset(get_relational_engine(), dataset_id, BEFORE)


async def _row(dataset_id) -> DatasetDatabase:
    async with get_relational_engine().get_async_session() as session:
        return await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
        )


async def _record(dataset_id) -> dict:
    return dict((await _row(dataset_id)).vector_database_connection_info)


async def _upgrade(dataset_id, version_changed: bool = False) -> dict | None:
    """What the runner does for one dataset on startup."""
    return await runner._migrate_dataset(
        get_relational_engine(), await _row(dataset_id), "test", version_changed, "head"
    )


async def _stage_with_the_local_default(dataset_id) -> None:
    """Stage the dataset's chunks as a re-embed does, and stop there."""
    async with DatabaseContextManager(dataset_id, check_embedding_model=False):
        # The cached engine embeds with the model that built the dataset.
        vector_engine_cache.evict(**get_vectordb_context_config())
        await stage_collections(await get_vector_engine_async(), ["_Chunk_text"])
        vector_engine_cache.evict(**get_vectordb_context_config())


async def _chunk_vectors(dataset_id, config: EmbeddingConfig | None = None) -> dict:
    """The stored vector of every chunk by its text, read as a user of the dataset does."""
    async with set_database_global_context_variables(dataset_id, embedding_config=config):
        store = await get_vector_engine_async()
        assert not [n for n in await store.get_collection_names() if n.startswith(STAGING_PREFIX)]
        rows = await store.retrieve(
            "_Chunk_text", await store.get_data_point_ids("_Chunk_text"), include_vector=True
        )
        return {row.payload["text"]: row.payload["vector"] for row in rows}


def _vectors_of(record: dict) -> dict:
    model = _Model(record[MODEL_KEY], record[DIMENSIONS_KEY])
    return {chunk.text: pytest.approx(model.vector(chunk.text)) for chunk in CHUNKS}


@pytest.mark.asyncio
async def test_a_dataset_of_the_previous_default_is_reembedded(dataset, caplog):
    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    # With the new default in force nothing but a migration gets into the dataset.
    with pytest.raises(EmbeddingDimensionMismatchError):
        async with set_database_global_context_variables(dataset.id):
            pass

    # As on a real upgrade the cognee version changed too, so the vector adapter's
    # storage sync runs after the chain, on the re-embedded store.
    with caplog.at_level("WARNING"):
        summary = await _upgrade(dataset.id, version_changed=True)

    assert summary["migrations_applied"] == [SLUG]
    assert (await _row(dataset.id)).migration_revision == SLUG
    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(LOCAL_DEFAULT)
    assert _Model.embedded.count(DEFAULT_LOCAL_EMBEDDING_MODEL) == len(CHUNKS)
    # The routing collection of the previous width is dropped before the dataset's
    # routes are written again.
    assert dataset.routing.calls == ["prune", "remove", "copy"]

    said = [r.getMessage() for r in caplog.records if "default local embedding" in r.getMessage()]
    assert len(said) == 1
    assert "changed from BAAI/bge-small-en-v1.5 to perplexity-ai/pplx-embed-v1-0.6b" in said[0]
    assert "should provide better accuracy" in said[0]
    assert "EMBEDDING_PROVIDER=fastembed and EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in said[0]

    # The dataset is written to and searched like any other one of the new default.
    async with set_database_global_context_variables(dataset.id):
        store = await get_vector_engine_async()
        await store.index_data_points("_Chunk", "text", [_Chunk(text="dddd")])
        found = await store.search("_Chunk_text", query_text="dddd", limit=1, include_payload=True)
        assert found[0].payload["text"] == "dddd"

    # A second run finds nothing to do.
    await runner._stamp_dataset(get_relational_engine(), dataset.id, BEFORE)
    await _upgrade(dataset.id)
    assert _Model.embedded.count(DEFAULT_LOCAL_EMBEDDING_MODEL) == len(CHUNKS) + 2
    assert dataset.routing.calls == ["prune", "remove", "copy"]


@pytest.mark.asyncio
async def test_a_routing_collection_of_the_new_width_is_kept(dataset):
    """The second dataset to be re-embedded must not drop the first one's routes."""
    dataset.routing.width_matches = True

    await _upgrade(dataset.id)

    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert dataset.routing.calls == ["remove", "copy"]


@pytest.mark.asyncio
async def test_a_dataset_with_nothing_stored_is_moved_too(dataset):
    """Its record names the previous default, and that alone keeps the new one out."""
    async with get_relational_engine().get_async_session() as session:
        empty = await create_dataset(f"empty_{uuid4().hex[:8]}", await get_default_user(), session)
        await session.commit()
    try:
        async with set_database_global_context_variables(empty.id, embedding_config=PREVIOUS):
            pass
        await runner._stamp_dataset(get_relational_engine(), empty.id, BEFORE)
        assert await _record(empty.id) == PREVIOUS_DEFAULT

        await _upgrade(empty.id)

        assert await _record(empty.id) == LOCAL_DEFAULT
        async with set_database_global_context_variables(empty.id):
            await (await get_vector_engine_async()).index_data_points("_Chunk", "text", CHUNKS)
        assert await _chunk_vectors(empty.id) == _vectors_of(LOCAL_DEFAULT)
    finally:
        await delete_dataset(empty)


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_key", [False, True], ids=["no_llm_key", "llm_key"])
async def test_a_configured_embedding_model_is_used_as_it_is(dataset, installation, llm_key):
    """Whoever set an embedding model keeps it, the previous default included."""
    installation(PREVIOUS, llm_key=llm_key)

    summary = await _upgrade(dataset.id)

    # The migration ran and had nothing to do, so the dataset is at head.
    assert summary["migrations_applied"] == [SLUG]
    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(PREVIOUS_DEFAULT)
    assert _Model.embedded == [PREVIOUS_DEFAULT[MODEL_KEY]] * len(CHUNKS)
    assert dataset.routing.calls == []


@pytest.mark.asyncio
async def test_an_llm_key_without_an_embedding_setting_is_not_the_local_default(
    dataset, installation
):
    """With a usable LLM key embeddings run on the OpenAI default, not on a local model."""
    installation(NOTHING_CONFIGURED, llm_key=True)

    await _upgrade(dataset.id)

    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    assert _Model.embedded == [PREVIOUS_DEFAULT[MODEL_KEY]] * len(CHUNKS)


@pytest.mark.asyncio
async def test_a_dataset_another_model_built_is_left_alone(dataset):
    async with get_relational_engine().get_async_session() as session:
        other = await create_dataset(f"other_{uuid4().hex[:8]}", await get_default_user(), session)
        await session.commit()
    try:
        await _build(other.id, OTHER)
        record = await _record(other.id)

        summary = await _upgrade(other.id)

        assert summary["migrations_applied"] == [SLUG]
        assert await _record(other.id) == record == {MODEL_KEY: "other", DIMENSIONS_KEY: 7}
        assert await _chunk_vectors(other.id, OTHER) == _vectors_of(record)
        assert DEFAULT_LOCAL_EMBEDDING_MODEL not in _Model.embedded
    finally:
        await delete_dataset(other)


@pytest.mark.asyncio
async def test_a_dataset_of_the_previous_width_that_names_no_model_says_it_is_left_alone(
    dataset, caplog
):
    """Built before the record named a model: which model built it is not known."""
    async with get_relational_engine().get_async_session() as session:
        row = await session.scalar(
            select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset.id)
        )
        row.vector_database_connection_info = {
            key: value
            for key, value in row.vector_database_connection_info.items()
            if key not in (MODEL_KEY, DIMENSIONS_KEY)
        }
        await session.commit()

    with caplog.at_level("WARNING"):
        summary = await _upgrade(dataset.id)

    assert summary["migrations_applied"] == [SLUG]
    assert DEFAULT_LOCAL_EMBEDDING_MODEL not in _Model.embedded
    said = [r.getMessage() for r in caplog.records if "does not name the model" in r.getMessage()]
    assert len(said) == 1
    assert str(dataset.id) in said[0] and "384-dimensional" in said[0]
    assert "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in said[0] and "reembed_dataset" in said[0]


@pytest.mark.asyncio
async def test_a_failed_reembed_says_how_to_keep_the_previous_model(dataset, installation):
    _Model.down.add(DEFAULT_LOCAL_EMBEDDING_MODEL)

    with pytest.raises(CogneeSystemError) as raised:
        await _upgrade(dataset.id)

    # What `cognee-cli current` shows of it: the way out, then the cause.
    kept = runner._error_text(raised.value)
    assert "set EMBEDDING_PROVIDER=fastembed and EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in kept
    assert "embedding provider is down" in kept
    # The dataset is as it was, and still before the migration.
    assert (await _row(dataset.id)).migration_revision == BEFORE
    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    assert await _chunk_vectors(dataset.id, PREVIOUS) == _vectors_of(PREVIOUS_DEFAULT)

    # With the previous model set, the next run has nothing to do and the dataset is used.
    installation(PREVIOUS)
    assert (await _upgrade(dataset.id))["migrations_applied"] == [SLUG]
    assert await _chunk_vectors(dataset.id) == _vectors_of(PREVIOUS_DEFAULT)


@pytest.mark.asyncio
async def test_a_reembed_that_was_interrupted_is_finished_by_the_next_run(dataset):
    """A process killed mid-way leaves the dataset marked; the runner still gets in."""
    await record_reembed_progress(await _row(dataset.id), LOCAL_DEFAULT)
    assert REEMBED_KEY in await _record(dataset.id)

    await _upgrade(dataset.id)

    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(LOCAL_DEFAULT)


@pytest.mark.asyncio
async def test_a_model_that_cannot_be_loaded_says_how_to_keep_the_previous_one(
    dataset, monkeypatch
):
    """An offline host: the routing check is the first thing that loads the model."""

    async def cannot_load(engine):
        raise ValueError("Could not load model perplexity-ai/pplx-embed-v1-0.6b from any source.")

    monkeypatch.setattr(routing, "routing_width_matches", cannot_load)

    with pytest.raises(CogneeSystemError) as raised:
        await _upgrade(dataset.id)

    kept = runner._error_text(raised.value)
    assert "is as it was. To keep the previous model, set EMBEDDING_PROVIDER=fastembed" in kept
    assert "Could not load model" in kept
    assert await _record(dataset.id) == PREVIOUS_DEFAULT


@pytest.mark.asyncio
async def test_a_reembed_that_fails_while_replacing_is_finished_by_the_next_run(dataset):
    """Its collections are of two widths then, so the previous model is no way out."""

    async def disk_full(vector_engine, collection_names):
        raise OSError("No space left on device")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(reembed_collections, "replace_collections", disk_full)
        with pytest.raises(CogneeSystemError) as raised:
            await _upgrade(dataset.id)

    kept = runner._error_text(raised.value)
    assert "is partly re-embedded" in kept and "Run the migrations again" in kept
    assert "To keep the previous model" not in kept
    assert "No space left on device" in kept
    assert REEMBED_KEY in await _record(dataset.id)
    assert (await _row(dataset.id)).migration_revision == BEFORE

    await _upgrade(dataset.id)

    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(LOCAL_DEFAULT)


@pytest.mark.asyncio
async def test_an_interrupted_reembed_is_undone_when_the_previous_model_is_set(
    dataset, installation
):
    """Ctrl-C while it embeds, then the setting the log names. The dataset must not be
    stamped as migrated while it still refuses every use."""
    await record_reembed_progress(await _row(dataset.id), LOCAL_DEFAULT)
    await _stage_with_the_local_default(dataset.id)
    installation(PREVIOUS)

    summary = await _upgrade(dataset.id)

    assert summary["migrations_applied"] == [SLUG]
    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    # Read as any user of the dataset reads it; no staging collection is left either.
    assert await _chunk_vectors(dataset.id) == _vectors_of(PREVIOUS_DEFAULT)


@pytest.mark.asyncio
async def test_a_half_replaced_dataset_is_not_stamped_when_the_previous_model_is_set(
    dataset, installation
):
    """It can only be finished with the local default, so the run fails and says so."""
    await _stage_with_the_local_default(dataset.id)
    await record_reembed_progress(
        await _row(dataset.id), {**LOCAL_DEFAULT, STAGED_COLLECTIONS_KEY: ["_Chunk_text"]}
    )
    installation(PREVIOUS)

    with pytest.raises(CogneeValidationError, match="can only be finished"):
        await _upgrade(dataset.id)
    assert (await _row(dataset.id)).migration_revision == BEFORE

    installation(NOTHING_CONFIGURED)
    await _upgrade(dataset.id)

    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(LOCAL_DEFAULT)


@pytest.mark.asyncio
async def test_a_callers_own_model_is_not_the_installations(dataset):
    """A failed run is retried by the next write, from inside that write's context. Its
    per-call model must not make the local default look configured away."""
    token = embedding_config_override.set(OTHER)
    try:
        await _upgrade(dataset.id)
    finally:
        embedding_config_override.reset(token)

    assert await _record(dataset.id) == LOCAL_DEFAULT
    assert await _chunk_vectors(dataset.id) == _vectors_of(LOCAL_DEFAULT)


@pytest.mark.asyncio
async def test_a_downgrade_reembeds_with_the_previous_default(dataset):
    await _upgrade(dataset.id)
    assert await _record(dataset.id) == LOCAL_DEFAULT

    summary = await runner._downgrade_dataset(
        get_relational_engine(), await _row(dataset.id), BEFORE
    )

    assert summary["migrations_reverted"] == [SLUG]
    assert (await _row(dataset.id)).migration_revision == BEFORE
    assert await _record(dataset.id) == PREVIOUS_DEFAULT
    assert await _chunk_vectors(dataset.id, PREVIOUS) == _vectors_of(PREVIOUS_DEFAULT)


class _SharedStore:
    """The one vector store of an installation with access control off."""

    def __init__(self, stored_width: int | None):
        self.stored_width = stored_width

    async def get_stored_vector_size(self) -> int | None:
        return self.stored_width


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_width", [None, 1024, 3072], ids=["empty", "new", "another"])
async def test_a_shared_store_is_left_alone(installation, stored_width):
    """Access control off: a new deployment, one already on the new default, another model's.
    None of them is refused (the migration raises for nothing else in this mode)."""
    assert (
        await migration.migrate(MigrationContext(None, _SharedStore(stored_width), dataset_id=None))
        is None
    )


@pytest.mark.asyncio
async def test_a_shared_store_of_the_previous_default_is_refused(installation):
    context = MigrationContext(None, _SharedStore(384), dataset_id=None)

    with pytest.raises(CogneeConfigurationError) as raised:
        await migration.migrate(context)

    # All of it fits what the runner keeps of a failure, the way out included.
    kept = runner._error_text(raised.value)
    assert "set EMBEDDING_PROVIDER=fastembed and EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" in kept
    assert kept.endswith("ingest it again.")

    # With the previous model set there is nothing to refuse.
    installation(PREVIOUS)
    await migration.migrate(context)
