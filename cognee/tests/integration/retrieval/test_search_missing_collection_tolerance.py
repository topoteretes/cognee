"""Multi-dataset search must survive datasets that lack a collection.

A dataset can be legitimately cognified (or simply added-but-not-cognified)
without carrying every standard vector collection — e.g. a pipeline route
that skips summarization never creates TextSummary_text. With per-dataset
databases (access control on, the default), CHUNKS/SUMMARIES/RAG searches
iterate every readable dataset; a missing collection used to raise
NoDataError ("No data found in the system") and abort the WHOLE merged
search, hiding the results of every healthy dataset. Caught live by the
backwards-compatibility CI when a DLT dataset broke the lorem searches.

Offline: LLM and embeddings are mocked.
"""

import pathlib
from unittest.mock import patch

import pytest
import pytest_asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.engine.operations.setup import setup as engine_setup

DOCS_DATASET = "docs_ds"
BARE_DATASET = "bare_ds"
DOC_TEXT = "Zorblatt Industries manufactures underwater bicycles in Reykjavik."


async def _mock_embed_text(self, texts):
    return [[float(len(t) % 7) / 10.0 + 0.1] * self.dimensions for t in texts]


async def _mock_llm(text_input, system_prompt, response_model, **kwargs):
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
        return KnowledgeGraph(
            nodes=[Node(id="zorblatt", name="Zorblatt", type="Company", description="mfg")],
            edges=[
                Edge(source_node_id="zorblatt", target_node_id="zorblatt", relationship_name="is")
            ],
        )
    if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
        return SummarizedContent(summary="Underwater bicycle maker.", description="")
    if response_model is str:
        return "mock answer"
    return response_model()


@pytest_asyncio.fixture
async def mixed_datasets(tmp_path, monkeypatch):
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    root = pathlib.Path(tmp_path)

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    with (
        patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text),
        patch.object(LLMGateway, "acreate_structured_output", new=_mock_llm),
    ):
        # A healthy dataset with every standard collection...
        await cognee.add(DOC_TEXT, dataset_name=DOCS_DATASET)
        await cognee.cognify(datasets=[DOCS_DATASET])
        # ...and a dataset that was added but never cognified: its vector
        # database exists with NO collections at all.
        await cognee.add("bare dataset content, never cognified", dataset_name=BARE_DATASET)
        yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_cross_dataset_search_survives_missing_collections(mixed_datasets):
    """Searches spanning both datasets return the healthy dataset's results."""
    chunk_results = await cognee.search(query_type=SearchType.CHUNKS, query_text="Zorblatt")
    assert chunk_results, "healthy dataset's chunks must survive the bare dataset"
    assert "Zorblatt" in str(chunk_results)

    summary_results = await cognee.search(query_type=SearchType.SUMMARIES, query_text="Zorblatt")
    assert summary_results, "healthy dataset's summaries must survive the bare dataset"

    rag_results = await cognee.search(query_type=SearchType.RAG_COMPLETION, query_text="Zorblatt")
    assert rag_results, "RAG completion must survive the bare dataset"

    lexical_results = await cognee.search(
        query_type=SearchType.CHUNKS_LEXICAL, query_text="Zorblatt"
    )
    assert lexical_results, "lexical chunk search must survive the bare dataset"


@pytest.mark.asyncio
async def test_search_on_collectionless_dataset_returns_empty(mixed_datasets):
    """A search scoped to ONLY the bare dataset yields empty results — not
    the misleading "No data found in the system" error."""
    results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text="anything", datasets=[BARE_DATASET]
    )
    # The per-dataset envelope survives; its search_result is simply empty.
    assert all(entry["search_result"] == [] for entry in results), results

    lexical = await cognee.search(
        query_type=SearchType.CHUNKS_LEXICAL, query_text="anything", datasets=[BARE_DATASET]
    )
    assert all(entry["search_result"] == [] for entry in lexical), lexical
