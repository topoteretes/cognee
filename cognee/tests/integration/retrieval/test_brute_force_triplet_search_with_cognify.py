import pathlib

import pytest
import pytest_asyncio
import cognee

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search


@pytest_asyncio.fixture
async def clean_environment():
    """Configure isolated storage and ensure cleanup before/after."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_brute_force_triplet_search_e2e")
    data_directory_path = str(base_dir / ".data_storage/test_brute_force_triplet_search_e2e")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_brute_force_triplet_search_end_to_end(clean_environment):
    """Minimal end-to-end exercise of single and batch triplet search."""

    text = """
        Cognee is an open-source AI memory engine that structures data into searchable formats for use with AI agents.
        The company focuses on persistent memory systems using knowledge graphs and vector search.
        It is a Berlin-based startup building infrastructure for context-aware AI applications.
        NLP systems can use Cognee to store and retrieve structured information.
    """

    await cognee.add(text)
    await cognee.cognify()

    single_result = await brute_force_triplet_search(
        query="What can NLP systems use Cognee for?",
        top_k=1,
    )
    assert isinstance(single_result, list)
    assert single_result
    assert all(isinstance(edge, Edge) for edge in single_result)

    batch_queries = ["What is Cognee?", "What is the company's focus?"]
    batch_result = await brute_force_triplet_search(query_batch=batch_queries, top_k=1)

    assert isinstance(batch_result, list)
    assert len(batch_result) == len(batch_queries)
    assert all(isinstance(per_query, list) for per_query in batch_result)
    assert all(per_query for per_query in batch_result)
    assert all(isinstance(edge, Edge) for per_query in batch_result for edge in per_query)
