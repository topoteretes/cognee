import pathlib
import os
import asyncio
import pytest
import pytest_asyncio
from collections import Counter

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.temporal_retriever import TemporalRetriever
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user

logger = get_logger()


async def _reset_engines_and_prune() -> None:
    """Reset db engine caches and prune data/system.

    Kept intentionally identical to the inlined setup logic to avoid event loop issues when
    using deployed databases (Neo4j, PostgreSQL) and to ensure fresh instances per run.
    """
    # Dispose of existing engines and clear caches to ensure fresh instances for each test
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = get_vector_engine()
        # Dispose SQLAlchemy engine connection pool if it exists
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        # Engine might not exist yet
        pass

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _seed_default_dataset(dataset_name: str) -> dict:
    """Add the shared test dataset contents and run cognify (same steps/order as before)."""
    text_1 = """Germany is located in europe right next to the Netherlands"""

    logger.info(f"Adding text data to dataset: {dataset_name}")
    await cognee.add(text_1, dataset_name)

    explanation_file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    logger.info(f"Adding file data to dataset: {dataset_name}")
    await cognee.add([explanation_file_path_quantum], dataset_name)

    logger.info(f"Running cognify on dataset: {dataset_name}")
    await cognee.cognify([dataset_name])

    return {
        "dataset_name": dataset_name,
        "text_1": text_1,
        "explanation_file_path_quantum": explanation_file_path_quantum,
    }


@pytest.fixture(scope="session")
def event_loop():
    """Use a single asyncio event loop for this test module.

    This helps avoid "Future attached to a different loop" when running multiple async
    tests that share clients/engines.
    """
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


async def setup_test_environment():
    """Helper function to set up test environment with data, cognify, and triplet embeddings."""
    # This test runs for multiple db settings, to run this locally set the corresponding db envs

    dataset_name = "test_dataset"
    logger.info("Starting test setup: pruning data and system")
    await _reset_engines_and_prune()
    state = await _seed_default_dataset(dataset_name=dataset_name)

    user = await get_default_user()
    from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings

    logger.info("Creating triplet embeddings")
    await create_triplet_embeddings(user=user, dataset=dataset_name, triplets_batch_size=5)

    # Check if Triplet_text collection was created
    vector_engine = get_vector_engine()
    has_collection = await vector_engine.has_collection(collection_name="Triplet_text")
    logger.info(f"Triplet_text collection exists after creation: {has_collection}")

    if has_collection:
        collection = await vector_engine.get_collection("Triplet_text")
        count = await collection.count_rows() if hasattr(collection, "count_rows") else "unknown"
        logger.info(f"Triplet_text collection row count: {count}")

    return state


async def _get_retriever_context(retriever, query: str):
    """Retrieve objects and resolve context via the retriever API."""
    retrieved_objects = await retriever.get_retrieved_objects(query)
    return await retriever.get_context_from_objects(query, retrieved_objects)


@pytest_asyncio.fixture(scope="session")
async def e2e_state():
    """Compute E2E artifacts once; tests only assert.

    This avoids repeating expensive setup and LLM calls across multiple tests.
    """
    await setup_test_environment()

    # --- Graph/vector engine consistency ---
    graph_engine = await get_graph_engine()
    _nodes, edges = await graph_engine.get_graph_data()

    vector_engine = get_vector_engine()
    collection = await vector_engine.search(
        collection_name="Triplet_text",
        query_text="Test",
        limit=None,
    )

    # --- Retriever contexts ---
    query = "Next to which country is Germany located?"

    contexts = {
        "graph_completion": await _get_retriever_context(GraphCompletionRetriever(), query=query),
        "graph_completion_cot": await _get_retriever_context(
            GraphCompletionCotRetriever(), query=query
        ),
        "graph_completion_context_extension": await _get_retriever_context(
            GraphCompletionContextExtensionRetriever(), query=query
        ),
        "graph_summary_completion": await _get_retriever_context(
            GraphSummaryCompletionRetriever(), query=query
        ),
        "chunks": await _get_retriever_context(ChunksRetriever(top_k=5), query=query),
        "summaries": await _get_retriever_context(SummariesRetriever(top_k=5), query=query),
        "rag_completion": await _get_retriever_context(CompletionRetriever(top_k=3), query=query),
        "temporal": await _get_retriever_context(TemporalRetriever(top_k=5), query=query),
        "triplet": await _get_retriever_context(TripletRetriever(), query=query),
    }

    # --- Retriever triplets + vector distance validation ---
    triplets = {
        "graph_completion": await GraphCompletionRetriever().get_triplets(query=query),
        "graph_completion_cot": await GraphCompletionCotRetriever().get_triplets(query=query),
        "graph_completion_context_extension": await GraphCompletionContextExtensionRetriever().get_triplets(
            query=query
        ),
        "graph_summary_completion": await GraphSummaryCompletionRetriever().get_triplets(
            query=query
        ),
    }

    # --- Search operations + graph side effects ---
    completion_gk = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where is germany located, next to which country?",
        save_interaction=True,
        verbose=True,
    )
    completion_cot = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_COT,
        query_text="What is the country next to germany??",
        save_interaction=True,
        verbose=True,
    )
    completion_ext = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        query_text="What is the name of the country next to germany",
        save_interaction=True,
        verbose=True,
    )

    completion_sum = await cognee.search(
        query_type=SearchType.GRAPH_SUMMARY_COMPLETION,
        query_text="Next to which country is Germany located?",
        save_interaction=True,
        verbose=True,
    )
    completion_triplet = await cognee.search(
        query_type=SearchType.TRIPLET_COMPLETION,
        query_text="Next to which country is Germany located?",
        save_interaction=True,
        verbose=True,
    )
    completion_chunks = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text="Germany",
        save_interaction=False,
        verbose=True,
    )
    completion_summaries = await cognee.search(
        query_type=SearchType.SUMMARIES,
        query_text="Germany",
        save_interaction=False,
        verbose=True,
    )
    completion_rag = await cognee.search(
        query_type=SearchType.RAG_COMPLETION,
        query_text="Next to which country is Germany located?",
        save_interaction=False,
        verbose=True,
    )
    completion_temporal = await cognee.search(
        query_type=SearchType.TEMPORAL,
        query_text="Next to which country is Germany located?",
        save_interaction=False,
        verbose=True,
    )

    # Snapshot after all E2E operations above (used by assertion-only tests).
    graph_snapshot = await (await get_graph_engine()).get_graph_data()

    return {
        "graph_edges": edges,
        "triplet_collection": collection,
        "vector_collection_edges_count": len(collection),
        "graph_edges_count": len(edges),
        "contexts": contexts,
        "triplets": triplets,
        "search_results": {
            "graph_completion": completion_gk,
            "graph_completion_cot": completion_cot,
            "graph_completion_context_extension": completion_ext,
            "graph_summary_completion": completion_sum,
            "triplet_completion": completion_triplet,
            "chunks": completion_chunks,
            "summaries": completion_summaries,
            "rag_completion": completion_rag,
            "temporal": completion_temporal,
        },
        "graph_snapshot": graph_snapshot,
    }


@pytest.mark.asyncio
async def test_e2e_graph_vector_consistency(e2e_state):
    """Graph and vector stores contain the same triplet edges."""
    assert e2e_state["graph_edges_count"] == e2e_state["vector_collection_edges_count"]


@pytest.mark.asyncio
async def test_e2e_retriever_contexts(e2e_state):
    """All retrievers return non-empty, well-typed contexts."""
    contexts = e2e_state["contexts"]

    for name in [
        "graph_completion",
        "graph_completion_cot",
        "graph_completion_context_extension",
        "graph_summary_completion",
    ]:
        ctx = contexts[name]
        assert isinstance(ctx, str), f"{name}: Context should be a string"
        assert ctx.strip(), f"{name}: Context should not be empty"
        lower = ctx.lower()
        assert "germany" in lower or "netherlands" in lower

    triplet_ctx = contexts["triplet"]
    assert isinstance(triplet_ctx, str), "triplet: Context should be a string"
    assert triplet_ctx.strip(), "triplet: Context should not be empty"

    chunks_ctx = contexts["chunks"]
    assert isinstance(chunks_ctx, str), "chunks: Context should be a string"
    assert chunks_ctx.strip(), "chunks: Context should not be empty"
    chunks_text = chunks_ctx.lower()
    assert "germany" in chunks_text or "netherlands" in chunks_text

    summaries_ctx = contexts["summaries"]
    assert isinstance(summaries_ctx, str), "summaries: Context should be a string"
    assert summaries_ctx.strip(), "summaries: Context should not be empty"

    rag_ctx = contexts["rag_completion"]
    assert isinstance(rag_ctx, str), "rag_completion: Context should be a string"
    assert rag_ctx.strip(), "rag_completion: Context should not be empty"

    temporal_ctx = contexts["temporal"]
    assert isinstance(temporal_ctx, str), "temporal: Context should be a string"
    assert temporal_ctx.strip(), "temporal: Context should not be empty"


@pytest.mark.asyncio
async def test_e2e_retriever_triplets_have_vector_distances(e2e_state):
    """Graph retriever triplets include sane vector_distance metadata."""
    for name, triplets in e2e_state["triplets"].items():
        assert isinstance(triplets, list), f"{name}: Triplets should be a list"
        assert triplets, f"{name}: Triplets list should not be empty"
        for edge in triplets:
            assert isinstance(edge, Edge), f"{name}: Elements should be Edge instances"
            vector_distances = edge.attributes.get("vector_distance")
            assert vector_distances is not None, (
                f"{name}: vector_distance should be set when retrievers return results"
            )
            assert isinstance(vector_distances, list) and vector_distances, (
                f"{name}: vector_distance should be a non-empty list"
            )
            distance = vector_distances[0]
            assert isinstance(distance, float), (
                f"{name}: vector_distance[0] should be float, got {type(distance)}"
            )
            assert 0 <= distance <= 1

            node1_distances = edge.node1.attributes.get("vector_distance")
            node2_distances = edge.node2.attributes.get("vector_distance")
            assert node1_distances is not None, (
                f"{name}: node1 vector_distance should be set when retrievers return results"
            )
            assert node2_distances is not None, (
                f"{name}: node2 vector_distance should be set when retrievers return results"
            )
            assert isinstance(node1_distances, list) and node1_distances, (
                f"{name}: node1 vector_distance should be a non-empty list"
            )
            assert isinstance(node2_distances, list) and node2_distances, (
                f"{name}: node2 vector_distance should be a non-empty list"
            )
            node1_distance = node1_distances[0]
            node2_distance = node2_distances[0]
            assert isinstance(node1_distance, float), (
                f"{name}: node1 vector_distance[0] should be float, got {type(node1_distance)}"
            )
            assert isinstance(node2_distance, float), (
                f"{name}: node2 vector_distance[0] should be float, got {type(node2_distance)}"
            )
            assert 0 <= node1_distance <= 1
            assert 0 <= node2_distance <= 1


@pytest.mark.asyncio
async def test_e2e_search_results_and_wrappers(e2e_state):
    """Search returns expected shapes across search types and access modes."""
    from cognee.context_global_variables import backend_access_control_enabled

    sr = e2e_state["search_results"]

    # Completion-like search types: validate wrapper + content
    for name in [
        "graph_completion",
        "graph_completion_cot",
        "graph_completion_context_extension",
        "graph_summary_completion",
        "triplet_completion",
        "rag_completion",
        "temporal",
    ]:
        search_results = sr[name]
        assert isinstance(search_results, list), f"{name}: should return a list"
        assert len(search_results) == 1, f"{name}: expected single-element list"

        if backend_access_control_enabled():
            wrapper = search_results[0]
            assert isinstance(wrapper, dict), (
                f"{name}: expected wrapper dict in access control mode"
            )
            assert wrapper.get("dataset_id"), f"{name}: missing dataset_id in wrapper"
            assert wrapper.get("dataset_name") == "test_dataset"
            result_payload = wrapper.get("text_result")
        else:
            entry = search_results[0]
            assert isinstance(entry, dict), f"{name}: expected dict entries"
            result_payload = entry.get("text_result")

        text_blob = str(result_payload)
        assert text_blob.strip()
        assert "netherlands" in text_blob.lower()

    # Non-LLM search types: CHUNKS / SUMMARIES validate payload list + text
    for name in ["chunks", "summaries"]:
        search_results = sr[name]
        assert isinstance(search_results, list), f"{name}: should return a list"
        assert search_results, f"{name}: should not be empty"

        entry = search_results[0]
        assert isinstance(entry, dict), f"{name}: expected dict entries"

        context_result = entry.get("context_result")
        text_result = entry.get("text_result")

        assert isinstance(context_result, str) and context_result.strip()
        lower_context = context_result.lower()
        assert "germany" in lower_context or "netherlands" in lower_context

        assert isinstance(text_result, list) and text_result
        first_text = text_result[0]
        assert isinstance(first_text, dict)
        assert str(first_text.get("text", "")).strip()


@pytest.mark.asyncio
async def test_e2e_graph_side_effects_and_node_fields(e2e_state):
    """Search interactions create expected graph nodes/edges and required fields."""
    graph = e2e_state["graph_snapshot"]
    nodes, edges = graph

    type_counts = Counter(node_data[1].get("type", {}) for node_data in nodes)
    edge_type_counts = Counter(edge_type[2] for edge_type in edges)

    assert type_counts.get("CogneeUserInteraction", 0) == 4
    assert type_counts.get("NodeSet", 0) == 1
    assert edge_type_counts.get("used_graph_element_to_answer", 0) >= 10
    assert edge_type_counts.get("belongs_to_set", 0) >= 4

    required_fields_user_interaction = {"question", "answer", "context"}

    for node_id, data in nodes:
        if data.get("type") == "CogneeUserInteraction":
            assert required_fields_user_interaction.issubset(data.keys())
            for field in required_fields_user_interaction:
                value = data[field]
                assert isinstance(value, str) and value.strip()
