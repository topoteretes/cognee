"""The routing candidate filter: which readable datasets the configured model can serve.

The registry is a real SQLite session; the configured model record is stubbed.
"""

import types
from importlib import import_module
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.search.dataset_routing import datasets_on_configured_model
from cognee.modules.users.models import DatasetDatabase

model_utils_module = import_module(
    "cognee.infrastructure.databases.utils.ensure_embedding_model_matches"
)

CONFIGURED = {model_utils_module.MODEL_KEY: "model-a", model_utils_module.DIMENSIONS_KEY: 4}


@pytest.fixture
def registry(monkeypatch):
    """A SQLite registry the rows are inserted into, with the configured model stubbed."""

    def fake_record():
        return dict(CONFIGURED)

    monkeypatch.setattr(model_utils_module, "embedding_model_record", fake_record)
    return get_relational_engine()


async def _add_dataset_row(dataset_id: UUID, info: dict | None) -> None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(
            insert(DatasetDatabase).values(
                dataset_id=dataset_id,
                owner_id=uuid4(),
                vector_database_name="v",
                graph_database_name="g",
                vector_database_provider="lancedb",
                graph_database_provider="kuzu",
                vector_dataset_database_handler="lancedb",
                graph_dataset_database_handler="kuzu",
                vector_database_connection_info=info or {},
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_a_dataset_recorded_with_the_configured_model_is_a_candidate(registry):
    dataset_id = uuid4()
    await _add_dataset_row(dataset_id, dict(CONFIGURED))

    assert await datasets_on_configured_model([dataset_id]) == {dataset_id}


@pytest.mark.asyncio
async def test_a_dataset_of_another_model_is_not_a_candidate_even_at_the_same_width(registry):
    """The model name is part of the record: two models of one width are still two
    models, and one's rows are never ranked against the other's query."""
    dataset_id = uuid4()
    await _add_dataset_row(
        dataset_id, {model_utils_module.MODEL_KEY: "model-b", model_utils_module.DIMENSIONS_KEY: 4}
    )

    assert await datasets_on_configured_model([dataset_id]) == set()


@pytest.mark.asyncio
async def test_a_dataset_recorded_at_another_width_is_not_a_candidate(registry):
    dataset_id = uuid4()
    await _add_dataset_row(dataset_id, dict(CONFIGURED, **{model_utils_module.DIMENSIONS_KEY: 8}))

    assert await datasets_on_configured_model([dataset_id]) == set()


@pytest.mark.asyncio
async def test_a_dataset_without_a_registry_row_is_a_candidate(registry):
    """Never entered a dataset context: no vectors, always searched."""
    dataset_id = uuid4()

    assert await datasets_on_configured_model([dataset_id]) == {dataset_id}


@pytest.mark.asyncio
async def test_a_legacy_row_recorded_at_the_configured_width_is_a_candidate(registry):
    """Before models were recorded, the row carries the width only: the width is
    what a collection fixes and what the model guard compares."""
    dataset_id = uuid4()
    await _add_dataset_row(
        dataset_id, {model_utils_module.MODEL_KEY: None, model_utils_module.DIMENSIONS_KEY: 4}
    )

    assert await datasets_on_configured_model([dataset_id]) == {dataset_id}


@pytest.mark.asyncio
async def test_a_legacy_row_recorded_at_another_width_is_not_a_candidate(registry):
    dataset_id = uuid4()
    await _add_dataset_row(
        dataset_id, {model_utils_module.MODEL_KEY: None, model_utils_module.DIMENSIONS_KEY: 8}
    )

    assert await datasets_on_configured_model([dataset_id]) == set()


@pytest.mark.asyncio
async def test_a_registry_read_failure_keeps_every_dataset(registry, monkeypatch, caplog):
    """The filter must never make a search worse: a read that fails searches
    everything, as before routing."""
    dataset_id = uuid4()

    class BrokenEngine:
        def get_async_session(self):
            raise RuntimeError("registry down")

    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine",
        lambda: BrokenEngine(),
    )

    with caplog.at_level("WARNING", logger="dataset_routing"):
        assert await datasets_on_configured_model([dataset_id]) == {dataset_id}
    assert "could not read the recorded embedding models" in caplog.text


@pytest.mark.asyncio
async def test_an_empty_ask_returns_without_touching_the_registry(monkeypatch):
    reads = []

    def fake_record():
        reads.append(1)
        return dict(CONFIGURED)

    monkeypatch.setattr(model_utils_module, "embedding_model_record", fake_record)
    assert await datasets_on_configured_model([]) == set()
    assert reads == []


@pytest.mark.asyncio
async def test_the_filter_resolves_the_deployment_model_not_a_per_call_override(
    registry, monkeypatch
):
    """The candidates are the datasets of the model the rows were mirrored under:
    a caller's per-call embedding override names no collection of its own."""
    from cognee.context_global_variables import embedding_config

    dataset_id = uuid4()
    await _add_dataset_row(dataset_id, dict(CONFIGURED))

    seen = []

    def fake_record():
        seen.append(embedding_config.get())
        return dict(CONFIGURED)

    monkeypatch.setattr(model_utils_module, "embedding_model_record", fake_record)
    override = types.SimpleNamespace(model="per-call-model")
    token = embedding_config.set(override)
    try:
        assert await datasets_on_configured_model([dataset_id]) == {dataset_id}
        assert embedding_config.get() is override
    finally:
        embedding_config.reset(token)

    assert seen == [None]
