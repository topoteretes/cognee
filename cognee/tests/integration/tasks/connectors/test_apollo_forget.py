# Adapted from topoteretes/cognee-community (Apache-2.0), PR #304, commit 83a980e.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Forget-on-delete must hard-delete a contact: its row, its Data record and its graph content.

Ingests two contacts, cognifies them with the LLM mocked, deletes one in Apollo and
checks that only the deleted contact leaves every store, rather than being replaced
by an emptied document.
"""

import importlib

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import apollo_source
from cognee.tests.unit.tasks.connectors.fake_apollo import FakeApollo

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "apollo_forget_test"
# Distinctive, unique tokens so each contact maps to exactly one graph entity.
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity named after whichever token appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        name = next((t for t in (ALPHA, BRAVO) if text_input and t in text_input), None)
        nodes = (
            [KGNode(id=name, name=name, type="Person", description=f"{name} entity")]
            if name
            else []
        )
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    token = token.lower()
    for _nid, props in nodes:
        if any(token in str(v).lower() for v in (props or {}).values()):
            return True
    return False


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
    }


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


async def _sync(fake: FakeApollo) -> None:
    await cognee.add(
        apollo_source(service=fake, include=("contacts",), resource_name="apollo_forget"),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


@pytest.mark.asyncio
async def test_deleting_a_contact_forgets_its_graph_content(clean_environment):
    fake = FakeApollo()
    fake.add_contact("c1", f"Ada {ALPHA}")
    fake.add_contact("c2", f"Ben {BRAVO}")

    await _sync(fake)
    assert await _graph_has(ALPHA), "the first contact should be in the graph after ingest"
    assert await _graph_has(BRAVO), "the second contact should be in the graph after ingest"
    before = await _documents()
    assert set(before) == {"contact:c1", "contact:c2"}

    del fake.contacts["c2"]
    await _sync(fake)

    assert await _documents() == {"contact:c1": before["contact:c1"]}, (
        "the deleted contact's Data record must go and the survivor must keep its id"
    )
    assert await _graph_has(ALPHA), "the surviving contact must stay in the graph"
    assert not await _graph_has(BRAVO), "the deleted contact's entity must leave the graph"


@pytest.mark.asyncio
async def test_forgetting_a_contact_by_hand_is_undone_by_the_next_sync(clean_environment):
    """The record is still in Apollo and still in the selection, so its staging row is still
    read back and its document is built again. Erasure has to go through Apollo or a filter."""
    fake = FakeApollo()
    fake.add_contact("c1", f"Ada {ALPHA}")
    fake.add_contact("c2", f"Ben {BRAVO}")
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before["contact:c2"], dataset=DATASET)
    assert set(await _documents()) == {"contact:c1"}
    assert not await _graph_has(BRAVO), "forget() must take the contact out of the graph"

    await _sync(fake)

    assert set(await _documents()) == {"contact:c1", "contact:c2"}
    assert await _graph_has(BRAVO), "the next sync builds the forgotten contact again"
