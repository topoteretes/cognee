import pathlib
import os
import pytest
import pytest_asyncio
import cognee
from cognee.context_global_variables import set_session_user_context_variable
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)
from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from collections import Counter

logger = get_logger()


@pytest_asyncio.fixture
async def setup_search_db_environment():
    """Set up test environment with data, cognify, and triplet embeddings."""
    # This test runs for multiple db settings, to run this locally set the corresponding db envs
    logger.info("Starting fixture setup: pruning data and system")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "test_dataset"

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

    graph_engine = await get_graph_engine()

    vector_engine = get_vector_engine()
    query = "Next to which country is Germany located?"
    gk_retriever_context = await GraphCompletionCotRetriever().get_context(query=query)
    gk_cot_retriever_context = await GraphCompletionCotRetriever().get_context(query=query)
    gk_ext_retriever_context = await GraphCompletionContextExtensionRetriever().get_context(
        query=query
    )
    gk_sum_retriever_context = await GraphSummaryCompletionRetriever().get_context(query=query)
    triplet_retriever_context = await TripletRetriever().get_context(query=query)

    yield {
        "dataset_name": dataset_name,
        "text_1": text_1,
        "explanation_file_path_quantum": explanation_file_path_quantum,
        "graph_engine": graph_engine,
        "vector_engine": vector_engine,
        "gk_retriever_context": gk_retriever_context,
        "gk_cot_retriever_context": gk_cot_retriever_context,
        "gk_ext_retriever_context": gk_ext_retriever_context,
        "gk_sum_retriever_context": gk_sum_retriever_context,
        "triplet_retriever_context": triplet_retriever_context,
    }

    logger.info("Fixture teardown: pruning data and system")
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest_asyncio.fixture
async def setup_search_db_environment_for_feedback():
    """Set up test environment for feedback weight calculation test."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "test_dataset"

    text_1 = """Germany is located in europe right next to the Netherlands"""
    await cognee.add(text_1, dataset_name)

    explanation_file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    await cognee.add([explanation_file_path_quantum], dataset_name)

    await cognee.cognify([dataset_name])

    yield {
        "dataset_name": dataset_name,
        "text_1": text_1,
        "explanation_file_path_quantum": explanation_file_path_quantum,
    }

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_graph_vector_engine_consistency(setup_search_db_environment):
    """Test that graph edges match triplet collection in vector engine."""
    vector_engine = setup_search_db_environment["vector_engine"]
    graph_engine = setup_search_db_environment["graph_engine"]

    nodes, edges = await graph_engine.get_graph_data()
    collection = await vector_engine.search(
        collection_name="Triplet_text", query_text="Test", limit=None
    )

    assert len(edges) == len(collection), (
        f"Expected {len(edges)} edges but got {len(collection)} in Triplet_text collection"
    )


@pytest.mark.asyncio
async def test_retriever_contexts(setup_search_db_environment):
    """Test that all retrievers return valid contexts with expected content."""

    context_gk = setup_search_db_environment["gk_retriever_context"]
    context_gk_cot = setup_search_db_environment["gk_cot_retriever_context"]
    context_gk_ext = setup_search_db_environment["gk_ext_retriever_context"]
    context_gk_sum = setup_search_db_environment["gk_sum_retriever_context"]
    context_triplet = setup_search_db_environment["triplet_retriever_context"]

    # Test graph-based retrievers (should return lists)
    for name, context in [
        ("GraphCompletionRetriever", context_gk),
        ("GraphCompletionCotRetriever", context_gk_cot),
        ("GraphCompletionContextExtensionRetriever", context_gk_ext),
        ("GraphSummaryCompletionRetriever", context_gk_sum),
    ]:
        assert isinstance(context, list), f"{name}: Context should be a list"
        assert len(context) > 0, f"{name}: Context should not be empty"

        context_text = await resolve_edges_to_text(context)
        lower = context_text.lower()
        assert "germany" in lower or "netherlands" in lower, (
            f"{name}: Context did not contain 'germany' or 'netherlands'; got: {context!r}"
        )

    # Test triplet retriever (should return string)
    assert isinstance(context_triplet, str), "TripletRetriever: Context should be a string"
    assert len(context_triplet) > 0, "TripletRetriever: Context should not be empty"
    lower_triplet = context_triplet.lower()
    assert "germany" in lower_triplet or "netherlands" in lower_triplet, (
        f"TripletRetriever: Context did not contain 'germany' or 'netherlands'; got: {context_triplet!r}"
    )
