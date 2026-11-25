"""What a Granola sync does to memory, with the LLM and embeddings mocked.

Each test follows a row of the contract's user scenario trace through ``cognee.remember`` the
way hosts call it: a note the reconcile proves gone (moved out, deleted) leaves the graph while
an entity another note names lives on, ``deleted_at`` does the same at once, a deselected scope
leaves memory, a forget by hand is undone by the next sync (the staging copy stays, an engine
gap), and the last test replays the scenario in one run.
"""

import importlib
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.integrations.ingestion import empty_resource
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import granola as granola_module
from cognee.tasks.ingestion.connectors import granola_source
from cognee.tests.unit.tasks.connectors.fake_granola import FakeGranola, note_id

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "granola_acme"
FOLDER = "fol_AAAAAAAAAAAAAA"
OTHER = "fol_BBBBBBBBBBBBBB"
TABLE = f"granola_notes_{FOLDER.lower()}"
# Distinctive tokens so each meeting maps to exactly one graph entity.
ALPHA = "Alphaproject"
BRAVO = "Bravoproject"
SHARED = "Sharedvendor"
TOKENS = (ALPHA, BRAVO, SHARED)


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        names = [t for t in TOKENS if text_input and t in text_input]
        nodes = [
            KGNode(id=name, name=name, type="Thing", description=f"{name} entity") for name in names
        ]
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


@pytest_asyncio.fixture
async def memory(clean_environment, monkeypatch):
    pytest.importorskip("ladybug")

    async def _noop_index(*_a, **_k):
        return None

    async def _mock_embed_text(self, text):
        return EmbeddingResult(vectors=[[0.0] * self.get_vector_size() for _ in text], tokens=0)

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)
    # Every completed walk reconciles, so a test needs no day to pass.
    monkeypatch.setattr(granola_module, "RECONCILE_INTERVAL", timedelta(0))
    yield


def _account(workspace_wide: bool = False) -> FakeGranola:
    fake = FakeGranola(workspace_wide=workspace_wide)
    fake.folder(FOLDER)
    fake.folder(OTHER)
    fake.clock = datetime.now(timezone.utc) - timedelta(hours=2)
    fake.add(1, folder=FOLDER, summary=f"{ALPHA} kickoff with {SHARED}")
    fake.add(2, folder=FOLDER, summary=f"{BRAVO} review with {SHARED}")
    return fake


def _now(fake: FakeGranola) -> None:
    fake.clock = datetime.now(timezone.utc)


async def _sync(fake, **kwargs):
    source = granola_source(scope=FOLDER, resource_name=TABLE, service=fake, **kwargs)
    await cognee.remember(
        source,
        dataset_name=DATASET,
        write_disposition="merge",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    return source


async def _dataset():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return user, (datasets[0] if datasets else None)


async def _documents() -> dict:
    _, dataset = await _dataset()
    if dataset is None:
        return {}
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(dataset.id)
        if isinstance(data.system_metadata, dict)
    }


async def _entity(name: str) -> bool:
    user, dataset = await _dataset()
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    return any(
        (props or {}).get("type") == "Entity"
        and str((props or {}).get("name", "")).lower() == name.lower()
        for _id, props in nodes
    )


@pytest.mark.asyncio
async def test_a_note_moved_out_of_the_folder_leaves_the_graph_and_a_shared_entity_stays(memory):
    fake = _account()
    await _sync(fake)
    assert await _entity(ALPHA) and await _entity(SHARED)

    _now(fake)
    fake.move(note_id(1), OTHER)
    source = await _sync(fake)

    assert set(await _documents()) == {note_id(2)}
    assert source.cognee_sync_stats["deleted"] == 1
    assert not await _entity(ALPHA), "the moved note's entity leaves with it"
    assert await _entity(SHARED), "the other note still names it"


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_wide", [False, True])
async def test_a_deleted_note_leaves_the_graph(memory, workspace_wide):
    fake = _account(workspace_wide)
    await _sync(fake)

    _now(fake)
    fake.delete(note_id(2))
    await _sync(fake)

    assert set(await _documents()) == {note_id(1)}
    assert not await _entity(BRAVO) and await _entity(SHARED)


@pytest.mark.asyncio
async def test_a_deselected_scope_leaves_memory(memory):
    await _sync(_account())

    await cognee.remember(
        empty_resource("granola", TABLE),
        dataset_name=DATASET,
        write_disposition="replace",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )

    assert await _documents() == {}
    assert not await _entity(ALPHA) and not await _entity(SHARED)


@pytest.mark.asyncio
async def test_forgetting_a_note_by_hand_is_undone_by_the_next_sync(memory):
    """It is still at Granola and its staging row is still read back. Erasure has to go
    through Granola or the selection."""
    fake = _account()
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before[note_id(1)], dataset=DATASET)
    assert note_id(1) not in await _documents()

    await _sync(fake)
    assert set(await _documents()) == set(before)


@pytest.mark.asyncio
async def test_the_scenario_replays_end_to_end(memory):
    """first sync, idle, edit, move out, empty listing kept, key replaced."""
    fake = _account()
    await _sync(fake)
    first = await _documents()
    assert set(first) == {note_id(1), note_id(2)}

    await _sync(fake)
    assert await _documents() == first

    _now(fake)
    fake.edit(note_id(2), summary_markdown=f"{BRAVO} postponed")
    await _sync(fake)
    assert (await _documents())[note_id(2)] != first[note_id(2)]

    fake.move(note_id(1), OTHER)
    await _sync(fake)
    assert set(await _documents()) == {note_id(2)} and not await _entity(ALPHA)

    fake.listing_hidden.update(fake.notes)
    source = await _sync(fake)
    assert source.cognee_sync_stats["failed_access"] == 1
    assert set(await _documents()) == {note_id(2)}
    fake.listing_hidden.clear()

    source = await _sync(fake, key_revision="key-2")
    assert source.cognee_sync_stats["deleted"] == 0
    assert set(await _documents()) == {note_id(2)} and await _entity(BRAVO)
