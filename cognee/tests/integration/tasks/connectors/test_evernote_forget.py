# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""What an Evernote sync does to memory, with the LLM and embeddings mocked.

A note trashed or expunged at Evernote leaves the graph while an entity another note names stays;
a note forgotten by hand comes back with the next sync, since it is still at Evernote and its
staging row is still read back.
"""

import importlib

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
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import evernote_source
from cognee.tests.unit.tasks.connectors.fake_evernote import FakeEvernote

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "evernote_forget_test"
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"
SHARED = "Sharedplace"
TOKENS = (ALPHA, BRAVO, SHARED)


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity per token that appears in the chunk text."""
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
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_a, **_k):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return EmbeddingResult(vectors=[[0.0] * self.get_vector_size() for _ in text], tokens=0)

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _account() -> FakeEvernote:
    fake = FakeEvernote()
    fake.add_notebook("nb1", "Journal")
    fake.add_note("alpha", f"{ALPHA} met us at {SHARED}.")
    fake.add_note("bravo", f"{BRAVO} lives near {SHARED}.")
    return fake


async def _sync(fake: FakeEvernote):
    source = evernote_source(resource_name="evernote_notes", service=fake)
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


async def _documents() -> dict[str, object]:
    _, dataset = await _dataset()
    if dataset is None:
        return {}
    return {
        str(data.system_metadata["external_id"]).removeprefix("note:"): data.id
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
@pytest.mark.parametrize("how", ["trash", "expunge"])
async def test_a_note_removed_at_evernote_leaves_the_graph_and_the_shared_entity_stays(
    clean_environment, how
):
    fake = _account()
    await _sync(fake)
    assert await _entity(ALPHA) and await _entity(BRAVO)

    fake.trash_note("alpha") if how == "trash" else fake.expunge_note("alpha")
    synced = await _sync(fake)

    assert synced.cognee_sync_stats["deleted"] == 1
    assert set(await _documents()) == {"bravo"}
    assert not await _entity(ALPHA), "the removed note's entity must leave the graph"
    assert await _entity(BRAVO) and await _entity(SHARED)


@pytest.mark.asyncio
async def test_forgetting_a_note_by_hand_is_undone_by_the_next_sync(clean_environment):
    """It is still at Evernote and its staging row is still read back. Erasure goes through
    Evernote or the selection."""
    fake = _account()
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before["alpha"], dataset=DATASET)
    assert "alpha" not in await _documents()

    await _sync(fake)

    assert set(await _documents()) == set(before)
    assert await _entity(ALPHA)
