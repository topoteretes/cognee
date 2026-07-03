"""End-to-end versioning proof through the REAL cognify pipeline (issue #3650).

The shared-fixture scenario the hackathon ticket asks for, deterministic in CI
via a mocked LLM (canned structured outputs) and hash-based embeddings — no
API keys, no network:

    add(doc1) -> cognify (run R1) -> snapshot "v1"
    add(doc2) -> cognify (run R2)
    - as-of "v1" returns exactly R1's artifacts (including the DocumentChunk
      vector-id set — the chunks/RAG surface);
    - rollback to "v1" restores R1's live state exactly; undo restores R2;
    - reversible forget of doc2's memory + undo restores exactly.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user
from cognee.modules.versioning import get_visible_artifacts_as_of, search_chunks_as_of

try:
    import ladybug  # noqa: F401
    import lancedb  # noqa: F401

    HAS_DEFAULT_STACK = True
except ModuleNotFoundError:
    HAS_DEFAULT_STACK = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_DEFAULT_STACK, reason="default stack not installed"),
]


def _hash_vector(text: str, size: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:size]]


async def _mock_structured_output(text_input: str, _system_prompt: str, response_model, **_kwargs):
    """Canned, deterministic structured outputs keyed on the input text."""
    from cognee.shared.data_models import Edge as KGEdge
    from cognee.shared.data_models import KnowledgeGraph, Node as KGNode, SummarizedContent

    if response_model is SummarizedContent:
        return SummarizedContent(summary="Deterministic summary", description="Mocked")

    if response_model is KnowledgeGraph:
        if "Alice" in text_input:
            return KnowledgeGraph(
                nodes=[
                    KGNode(id="Alice", name="Alice", type="Person", description="Person A"),
                    KGNode(id="Bob", name="Bob", type="Person", description="Person B"),
                ],
                edges=[
                    KGEdge(source_node_id="Alice", target_node_id="Bob", relationship_name="knows")
                ],
            )
        return KnowledgeGraph(
            nodes=[
                KGNode(id="Carol", name="Carol", type="Person", description="Person C"),
            ],
            edges=[],
        )

    if response_model is str:
        return "mocked"

    # Generic fallback: minimal valid instance for any other model.
    try:
        return response_model()
    except Exception:
        return response_model.model_construct()


@pytest_asyncio.fixture
async def e2e_env(request, tmp_path, monkeypatch):
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "ladybug")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")

    root = pathlib.Path(tmp_path) / request.node.name

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
    cognee.config.set_graph_db_config(
        {"graph_database_provider": "ladybug", "graph_dataset_database_handler": "ladybug"}
    )
    cognee.config.set_vector_db_config(
        {"vector_db_provider": "lancedb", "vector_dataset_database_handler": "lancedb"}
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    async def _fake_embed_text(self, texts):
        return [_hash_vector(text) for text in texts]

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _fake_embed_text)
    monkeypatch.setattr(LiteLLMEmbeddingEngine, "get_vector_size", lambda self: 8)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


async def _all_node_ids(graph) -> set[str]:
    rows = await graph.query("MATCH (n:Node) RETURN n.id", {})
    return {row[0] for row in rows}


async def _chunk_ids(vector) -> set[str]:
    hits = await vector.search("DocumentChunk_text", query_text="anything", limit=None)
    return {str(hit.id) for hit in hits}


async def test_snapshot_mutate_time_travel_rollback_and_undo_forget(e2e_env):
    user = await get_default_user()
    dataset_name = "versioning_e2e_dataset"

    # --- R1: first document through the real pipeline (mocked LLM). ---------
    await cognee.add("Alice knows Bob since 2020.", dataset_name=dataset_name, user=user)
    cognify_1 = await cognee.cognify(datasets=[dataset_name], user=user)
    (dataset_id,) = cognify_1.keys()
    assert cognify_1[dataset_id].status == "PipelineRunCompleted"

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        r1_node_ids = await _all_node_ids(graph)
        r1_chunk_ids = await _chunk_ids(vector)
    assert r1_chunk_ids, "cognify should have produced document chunks"

    # --- snapshot, then R2: second document. ---------------------------------
    await cognee.snapshot("v1", dataset_id=dataset_id, user=user)

    add_2 = await cognee.add("Carol appears later.", dataset_name=dataset_name, user=user)
    data_id_2 = add_2.data_ingestion_info[0]["data_id"]
    cognify_2 = await cognee.cognify(datasets=[dataset_name], user=user)
    assert cognify_2[dataset_id].status == "PipelineRunCompleted"

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        live_node_ids = await _all_node_ids(graph)
        live_chunk_ids = await _chunk_ids(vector)

        assert r1_node_ids < live_node_ids  # R2 added artifacts
        assert r1_chunk_ids < live_chunk_ids

        # --- time-travel: as-of "v1" is exactly the R1 state. ----------------
        visible_nodes, _ = await get_visible_artifacts_as_of(graph, dataset_id, "v1")
        assert visible_nodes == r1_node_ids

        # The chunks surface (user-facing RAG reads) filters to the exact
        # R1 chunk-vector id set — not merely a subset.
        as_of_hits = await search_chunks_as_of(
            graph, vector, dataset_id, "anything", "v1", top_k=50
        )
        assert {str(hit.id) for hit in as_of_hits} == r1_chunk_ids

    # --- rollback to "v1", then undo the rollback. ---------------------------
    rollback_result = await cognee.rollback("v1", dataset_id=dataset_id, user=user)
    assert rollback_result["status"] == "success"
    assert rollback_result["rolled_back_runs"], "R2 should have been reversed"

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        assert await _all_node_ids(graph) == r1_node_ids
        assert await _chunk_ids(vector) == r1_chunk_ids

    await cognee.undo(rollback_result["operation_id"], dataset_id=dataset_id, user=user)

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        assert await _all_node_ids(graph) == live_node_ids
        assert await _chunk_ids(vector) == live_chunk_ids

    # --- reversible forget of doc2's memory, then undo. ----------------------
    forget_result = await cognee.forget(
        data_id=data_id_2,
        dataset_id=dataset_id,
        memory_only=True,
        reversible=True,
        user=user,
    )
    assert forget_result["status"] == "success"

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        node_ids_after_forget = await _all_node_ids(graph)
        assert node_ids_after_forget < live_node_ids  # something was forgotten
        assert r1_node_ids <= node_ids_after_forget  # R1's memory untouched

    await cognee.undo(forget_result["operation_id"], dataset_id=dataset_id, user=user)

    async with set_database_global_context_variables(dataset_id, user.id):
        graph = await get_graph_engine()
        vector = get_vector_engine()
        # DO NOT REMOVE / weaken: full-state equality is the capture<->delete
        # invariant. Reversible forget is correct only if capture
        # (find_{node,edge}_source_refs_by_dataset) discovers exactly what
        # delete (delete_dataset_nodes_and_edges) removes. If the delete path
        # ever changes, this equality is what catches silent undo loss.
        assert await _all_node_ids(graph) == live_node_ids
        assert await _chunk_ids(vector) == live_chunk_ids
