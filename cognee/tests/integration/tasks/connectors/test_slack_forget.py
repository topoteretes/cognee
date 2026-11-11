"""Forget-on-delete through real add + cognify: a conversation deleted in Slack leaves memory.

Two conversations are ingested and cognified with the LLM mocked; each names its own person
and both name one shared project. Deleting one root in Slack and syncing must remove its
document and its own entity, and keep the shared entity, which the other document still owns.
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
from cognee.tasks.ingestion.connectors import slack_source
from cognee.tests.unit.tasks.connectors.fake_slack import CHANNEL, TEAM, FakeSlack

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "slack_forget_test"
# Distinctive tokens so each maps to exactly one graph entity.
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"
SHARED = "Sharedproject"
TOKENS = (ALPHA, BRAVO, SHARED)
ROOT_A = "1793872860.000100"
REPLY_A = "1793872920.000100"
ROOT_B = "1793872980.000100"
# ROOT_B has no replies, so its document is its day's (G7).
DAY_B = f"{CHANNEL}:day:2026-11-05"


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity for every token that appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        names = [token for token in TOKENS if text_input and token in text_input]
        return KnowledgeGraph(
            nodes=[KGNode(id=n, name=n, type="Thing", description=f"{n} entity") for n in names],
            edges=[],
        )
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


async def _dataset():
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return user, dataset


async def _documents() -> dict[str, object]:
    _, dataset = await _dataset()
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(dataset.id)
        if isinstance(data.system_metadata, dict)
    }


async def _entities() -> set[str]:
    user, dataset = await _dataset()
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
    names = {str((props or {}).get("name", "")).lower() for _, props in nodes}
    return {token for token in TOKENS if token.lower() in names}


async def _sync(fake: FakeSlack) -> None:
    await cognee.add(
        slack_source(
            service=fake, team_id=TEAM, channel_id=CHANNEL, resource_name="slack_channel_forget"
        ),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


def _channel() -> FakeSlack:
    fake = FakeSlack()
    fake.add_user("U1", "ana")
    fake.add_user("U2", "ben")
    fake.post(ROOT_A, "U1", f"{ALPHA} owns {SHARED}.")
    fake.reply(ROOT_A, REPLY_A, "U2", "Agreed.")
    fake.post(ROOT_B, "U2", f"{BRAVO} reviews {SHARED}.")
    return fake


@pytest.mark.asyncio
async def test_a_root_deleted_in_slack_leaves_memory_and_a_shared_entity_stays(
    clean_environment,
):
    fake = _channel()
    await _sync(fake)
    before = await _documents()
    assert set(before) == {f"{CHANNEL}:{ROOT_A}", DAY_B}
    assert await _entities() == {ALPHA, BRAVO, SHARED}

    fake.delete(ROOT_B)
    await _sync(fake)

    assert await _documents() == {f"{CHANNEL}:{ROOT_A}": before[f"{CHANNEL}:{ROOT_A}"]}, (
        "the deleted conversation's Data record goes and the survivor keeps its id"
    )
    assert await _entities() == {ALPHA, SHARED}, (
        "the deleted conversation's own entity goes; the one the other document names stays"
    )


@pytest.mark.asyncio
async def test_forgetting_a_conversation_by_hand_is_undone_by_the_next_sync(clean_environment):
    """The root is still in Slack, so its staging row is still read back and its document is
    built again. Erasure has to go through Slack or the channel selection."""
    fake = _channel()
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before[DAY_B], dataset=DATASET)
    assert set(await _documents()) == {f"{CHANNEL}:{ROOT_A}"}
    assert BRAVO not in await _entities()

    await _sync(fake)

    assert set(await _documents()) == set(before)
    assert BRAVO in await _entities()


@pytest.mark.asyncio
async def test_forgetting_only_the_memory_keeps_the_documents_and_the_next_sync_rebuilds_it(
    clean_environment,
):
    """``forget(memory_only=True)`` drops graph and vectors and keeps the data records; the
    next sync cognifies them again, so the rebuild needs no new Slack reads."""
    fake = _channel()
    await _sync(fake)
    before = set(await _documents())

    await cognee.forget(dataset=DATASET, memory_only=True)
    assert set(await _documents()) == before
    assert await _entities() == set()

    await _sync(fake)

    assert set(await _documents()) == before
    assert await _entities() == set(TOKENS)


@pytest.mark.asyncio
async def test_a_pruned_system_is_rebuilt_by_the_next_sync_from_the_staging_copy(
    clean_environment,
):
    """``prune_system`` drops cognee's stores, not dlt's staging copy and state, so the next
    sync reads the staged rows back and builds every document again (B1, SDK-910; accepted
    for Slack 2026-11-10 with ``forget``)."""
    fake = _channel()
    await _sync(fake)
    before = set(await _documents())

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await _sync(fake)

    assert set(await _documents()) == before
    assert await _entities() == set(TOKENS)
