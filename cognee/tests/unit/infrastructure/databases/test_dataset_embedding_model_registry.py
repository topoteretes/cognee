"""A dataset remembers the embedding model that built it, and refuses another one.

The registry row's ``vector_database_connection_info`` records the model at
creation; the dataset context compares it against the configured model once
per entry. Rows from before the keys existed are completed once from the store
itself. No call to the vector store on the happy path.
"""

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

import cognee.infrastructure.databases.utils.ensure_embedding_model_matches as ensure_mod
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
    DIMENSIONS_KEY,
    MODEL_KEY,
    clear_embedding_model_records,
    embedding_model_record,
    ensure_embedding_model_matches,
)
from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
    get_or_create_dataset_database,
)
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.databases.vector.exceptions import EmbeddingDimensionMismatchError
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.llm.config import LLMConfig
from cognee.modules.data.methods import create_dataset
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase

preflight_mod = importlib.import_module("cognee.modules.preflight.config_preflight")

SMALL = {MODEL_KEY: "BAAI/bge-small-en-v1.5", DIMENSIONS_KEY: 384}
LARGE = {MODEL_KEY: "openai/text-embedding-3-large", DIMENSIONS_KEY: 3072}


def _row(**info):
    return SimpleNamespace(
        dataset_id=uuid4(), vector_database_connection_info={"host": "h", **info}
    )


@pytest.fixture
def store(monkeypatch):
    """The dataset's vector engine, as the check sees it. ``None`` = no store reader."""
    state = {"engine": None}

    async def _engine():
        return state["engine"]

    monkeypatch.setattr(ensure_mod, "get_vector_engine_async", _engine)
    monkeypatch.setattr(ensure_mod, "_record_embedding_model", AsyncMock())
    # What the engine would embed with for these tests; the keyless resolution
    # itself is covered by test_keyless_record_follows_the_engine_not_the_config.
    monkeypatch.setattr(ensure_mod, "embedding_model_record", lambda: dict(LARGE))
    return state


@pytest.mark.asyncio
async def test_same_width_passes_without_touching_the_store(store):
    """The row already records what the engine embeds with: no store call, no write."""
    store["engine"] = SimpleNamespace(get_stored_vector_size=AsyncMock(return_value=384))

    await ensure_embedding_model_matches(_row(**LARGE))

    store["engine"].get_stored_vector_size.assert_not_awaited()
    ensure_mod._record_embedding_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_different_width_is_refused_with_both_models_named(store):
    row = _row(**SMALL)

    with pytest.raises(EmbeddingDimensionMismatchError) as raised:
        await ensure_embedding_model_matches(row)

    error = raised.value
    assert error.status_code == 409
    assert str(row.dataset_id) in error.message
    assert "BAAI/bge-small-en-v1.5" in error.message and "384" in error.message
    assert "openai/text-embedding-3-large" in error.message and "3072" in error.message
    assert "EMBEDDING_MODEL" in error.remediation and "forget" in error.remediation


@pytest.mark.asyncio
async def test_a_legacy_row_is_completed_from_the_store_once_then_checked(store):
    """A dev-era row has no keys: the width comes from the store, is recorded,
    and the configured model is checked against it in the same entry."""
    store["engine"] = SimpleNamespace(get_stored_vector_size=AsyncMock(return_value=384))
    row = _row()

    with pytest.raises(EmbeddingDimensionMismatchError) as raised:
        await ensure_embedding_model_matches(row)

    store["engine"].get_stored_vector_size.assert_awaited_once()
    ensure_mod._record_embedding_model.assert_awaited_once_with(
        row, {MODEL_KEY: None, DIMENSIONS_KEY: 384}
    )
    # The model that built a legacy row is unknown; the message still says the width.
    assert "an earlier embedding model (384 dimensions)" in raised.value.message
    assert "384-dimensional model" in raised.value.remediation


@pytest.mark.asyncio
async def test_a_legacy_row_over_an_empty_store_adopts_the_configured_model(store):
    store["engine"] = SimpleNamespace(get_stored_vector_size=AsyncMock(return_value=None))
    row = _row()

    await ensure_embedding_model_matches(row)

    ensure_mod._record_embedding_model.assert_awaited_once_with(row, LARGE)


@pytest.mark.asyncio
async def test_a_store_without_a_width_reader_is_left_alone(store):
    """Providers other than LanceDB and pgvector: no record, no refusal."""
    store["engine"] = SimpleNamespace()

    await ensure_embedding_model_matches(_row())

    ensure_mod._record_embedding_model.assert_not_awaited()


def _lancedb_adapter(tables):
    import pyarrow as pa

    class _Table:
        def __init__(self, width):
            self._schema = (
                pa.schema([("id", pa.string())])
                if width is None
                else pa.schema([("vector", pa.list_(pa.float32(), width))])
            )

        async def schema(self):
            return self._schema

    opened = {name: _Table(width) for name, width in tables.items()}

    class _Connection:
        async def table_names(self):
            return list(opened)

        async def open_table(self, name):
            return opened[name]

    adapter = LanceDBAdapter.__new__(LanceDBAdapter)
    adapter.get_connection = AsyncMock(return_value=_Connection())
    return adapter


@pytest.mark.asyncio
async def test_lancedb_reads_the_width_from_the_vector_tables():
    adapter = _lancedb_adapter({"meta": None, "Entity_name": 384})
    assert await adapter.get_stored_vector_size() == 384

    assert await _lancedb_adapter({}).get_stored_vector_size() is None
    assert await _lancedb_adapter({"meta": None}).get_stored_vector_size() is None


@pytest.mark.asyncio
async def test_lancedb_mixed_widths_do_not_depend_on_table_order(caplog):
    """A store built across a model change holds two widths.

    Whichever is recorded is wrong for some collections, so the answer must at
    least be the same every time and be reported — not whichever table the
    store happens to list first.
    """
    tables = {"Entity_name": 384, "DocumentChunk_text": 384, "NewType_name": 3072}
    reversed_tables = dict(reversed(list(tables.items())))

    with caplog.at_level(logging.WARNING):
        assert await _lancedb_adapter(tables).get_stored_vector_size() == 384
        assert await _lancedb_adapter(reversed_tables).get_stored_vector_size() == 384

    assert "different vector widths" in caplog.text
    assert "{384: 2, 3072: 1}" in caplog.text


@pytest.mark.asyncio
async def test_lancedb_ties_resolve_to_the_smaller_width():
    assert await _lancedb_adapter({"a": 3072, "b": 384}).get_stored_vector_size() == 384


@pytest.mark.asyncio
async def test_one_width_logs_nothing(caplog):
    """The warning must stay rare: a consistent store is the normal case."""
    with caplog.at_level(logging.WARNING):
        assert await _lancedb_adapter({"a": 384, "b": 384}).get_stored_vector_size() == 384

    assert caplog.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (["vector(1536)"], 1536),
        (["vector"], None),
        ([], None),
        # Built across a model change: the majority width, whatever the catalog order.
        (["vector(384)", "vector(3072)", "vector(384)"], 384),
        (["vector(3072)", "vector(384)", "vector(384)"], 384),
    ],
)
async def test_pgvector_reads_the_width_from_the_catalog(declared, expected):
    # The postgres extra is optional; the OS-matrix unit jobs run without it.
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from contextlib import asynccontextmanager

    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    recorded = {}

    class _Result:
        def scalars(self):
            return SimpleNamespace(all=lambda: declared)

    class _Session:
        async def execute(self, statement, parameters):
            recorded["parameters"] = parameters
            return _Result()

    @asynccontextmanager
    async def _session():
        yield _Session()

    adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    adapter.schema = "dataset_42"
    adapter.get_async_session = _session

    assert await adapter.get_stored_vector_size() == expected
    assert recorded["parameters"] == {"schema": "dataset_42"}


@pytest.mark.asyncio
async def test_creation_records_the_model_and_prune_clears_it(monkeypatch):
    """End to end on the real registry: the row carries the model that built the
    dataset inside its connection info, and clearing (what a vector prune does)
    removes only those keys."""
    # Module object, not the package's re-export of the function under the same name.
    create_mod = importlib.import_module(
        "cognee.infrastructure.databases.utils.get_or_create_dataset_database"
    )
    monkeypatch.setattr(create_mod, "embedding_model_record", lambda: dict(SMALL))
    # Pin the configured side too, so the switch below is deterministic in any
    # environment -- ambient keys and EMBEDDING_* settings change what a real
    # resolution returns (keyless reroutes to the local model).
    monkeypatch.setattr(ensure_mod, "embedding_model_record", lambda: dict(LARGE))
    user = await get_default_user()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        dataset = await create_dataset(f"embed_model_{uuid4().hex[:8]}", user, session)
        await session.commit()
        dataset_id = dataset.id

    async def _info():
        async with engine.get_async_session() as session:
            row = await session.scalar(
                select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
            )
            return dict(row.vector_database_connection_info)

    try:
        row = await get_or_create_dataset_database(dataset_id, user)
        assert row.vector_database_connection_info[MODEL_KEY] == SMALL[MODEL_KEY]
        assert row.vector_database_connection_info[DIMENSIONS_KEY] == 384
        other_keys = {
            k for k in row.vector_database_connection_info if k not in (MODEL_KEY, DIMENSIONS_KEY)
        }

        with pytest.raises(EmbeddingDimensionMismatchError):
            await ensure_embedding_model_matches(row)

        await clear_embedding_model_records()
        cleared = await _info()
        assert MODEL_KEY not in cleared and DIMENSIONS_KEY not in cleared
        assert set(cleared) == other_keys  # nothing else was touched

        # A runtime-only value must never be persisted by the record step
        # (pgvector injects credentials into this dict after loading).
        row.vector_database_connection_info = {**cleared, "password": "runtime-secret"}
        await ensure_mod._record_embedding_model(row, {MODEL_KEY: "m", DIMENSIONS_KEY: 7})
        persisted = await _info()
        assert persisted[DIMENSIONS_KEY] == 7
        assert "password" not in persisted
    finally:
        async with engine.get_async_session() as session:
            await session.execute(
                delete(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
            )
            await session.execute(delete(Dataset).where(Dataset.id == dataset_id))
            await session.commit()


@pytest.mark.parametrize(
    ("llm_api_key", "expected"),
    [
        (None, SMALL),  # keyless: embeddings are rerouted to the local model
        ("sk-a-real-key", LARGE),  # a usable key keeps the configured embedder
    ],
)
def test_keyless_record_follows_the_engine_not_the_config(monkeypatch, llm_api_key, expected):
    """The record must be what the ENGINE embeds with, not what the config says.

    ``get_embedding_engine`` resolves through ``resolve_embedding_defaults``,
    which on a keyless install (no embedding settings, no usable LLM key)
    reroutes to the local fastembed model. The raw config still reads as the
    stock openai/text-embedding-3-large (3072) that nothing embeds with, so
    recording the config would both false-positive on keyless deployments and
    miss the real mismatch when a key is later added.

    Real config objects on purpose: stand-ins cannot show this divergence.
    """
    # Hermetic: EmbeddingConfig is pydantic-settings, so ambient EMBEDDING_*
    # variables (CI sets EMBEDDING_DIMENSIONS=300) and a local .env would
    # otherwise leak into the "nothing configured" config this test needs.
    for variable in (
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_MODEL",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_API_KEY",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_API_BASE",
    ):
        monkeypatch.delenv(variable, raising=False)
    embedding_config = EmbeddingConfig(
        _env_file=None,
        embedding_provider="openai",
        embedding_model="openai/text-embedding-3-large",
        embedding_api_key=None,
        embedding_endpoint=None,
    )
    llm_config = LLMConfig(
        llm_provider="openai", llm_model="openai/gpt-5-mini", llm_api_key=llm_api_key or ""
    )
    monkeypatch.setattr(ensure_mod, "get_embedding_context_config", lambda: embedding_config)
    monkeypatch.setattr(ensure_mod, "get_llm_context_config", lambda: llm_config)
    monkeypatch.setattr(preflight_mod, "_skip_preflight", lambda: False)

    assert embedding_model_record() == expected
    # The config alone would have said 3072 in both cases.
    assert embedding_config.embedding_dimensions == 3072
