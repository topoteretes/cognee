"""Forget-on-delete must hard-delete an article: its row, its Data record and its graph content.

Ingests two articles, cognifies them with the LLM mocked, removes one from the PubMed
index and checks that only the removed article leaves every store, rather than being
replaced by an emptied document.
"""

import importlib

import cognee
import pytest
import pytest_asyncio
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from fake_pubmed import FakePubMed

from cognee_connector_pubmed import pubmed_source

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "pubmed_forget_test"
END = "2026/06/01"
# Distinctive, unique tokens so each article maps to exactly one graph entity.
ALPHA = "Alphastudy"
BRAVO = "Bravostudy"


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
            [KGNode(id=name, name=name, type="Study", description=f"{name} entity")] if name else []
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


async def _sync(fake: FakePubMed) -> None:
    await cognee.add(
        pubmed_source(service=fake, resource_name="pubmed_forget", end_date=END),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


@pytest.mark.asyncio
async def test_removing_an_article_forgets_its_graph_content(clean_environment):
    fake = FakePubMed()
    fake.add_article("100", f"Sleep and {ALPHA}")
    fake.add_article("101", f"Amyloid and {BRAVO}")

    await _sync(fake)
    assert await _graph_has(ALPHA), "the first article should be in the graph after ingest"
    assert await _graph_has(BRAVO), "the second article should be in the graph after ingest"
    before = await _documents()
    assert set(before) == {"pmid:100", "pmid:101"}

    fake.removed.add("101")
    await _sync(fake)

    assert await _documents() == {"pmid:100": before["pmid:100"]}, (
        "the removed article's Data record must go and the survivor must keep its id"
    )
    assert await _graph_has(ALPHA), "the surviving article must stay in the graph"
    assert not await _graph_has(BRAVO), "the removed article's entity must leave the graph"


@pytest.mark.asyncio
async def test_forgetting_an_article_by_hand_is_undone_by_the_next_sync(clean_environment):
    """The record is still in the index, so its staging row is still read back and its
    document is built again. Erasure has to go through the index or the window."""
    fake = FakePubMed()
    fake.add_article("100", f"Sleep and {ALPHA}")
    fake.add_article("101", f"Amyloid and {BRAVO}")
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before["pmid:101"], dataset=DATASET)
    assert set(await _documents()) == {"pmid:100"}
    assert not await _graph_has(BRAVO), "forget() must take the article out of the graph"

    await _sync(fake)

    assert set(await _documents()) == {"pmid:100", "pmid:101"}
    assert await _graph_has(BRAVO), "the next sync builds the forgotten article again"
