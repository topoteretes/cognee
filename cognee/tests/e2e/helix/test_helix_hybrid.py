"""End-to-end test for the HelixDB unified backend.

Exercises add -> cognify -> search with GRAPH_DATABASE_PROVIDER=helix and
VECTOR_DB_PROVIDER=helix, so a single HelixDB instance backs both the graph and
vector stores via HelixHybridAdapter.

Requires (otherwise the test self-skips):
  - A reachable HelixDB v2 gateway. Start one locally with:
        docker run -p 6969:6969 ghcr.io/helixdb/database-dev
    Override the URL with HELIX_TEST_URL (default http://localhost:6969).
  - LLM_API_KEY set for cognify.

Run directly: ``python -m cognee.tests.e2e.helix.test_helix_hybrid``
or via pytest (skips automatically when prerequisites are missing).
"""

import os
import asyncio

import httpx
import pytest

HELIX_URL = os.environ.get("HELIX_TEST_URL", "http://localhost:6969")


def _helix_reachable() -> bool:
    try:
        # A malformed body still proves the gateway is listening (non-connection error).
        httpx.post(f"{HELIX_URL}/v1/query", json={}, timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


async def main():
    # Import cognee first -- its __init__ calls dotenv.load_dotenv(override=True),
    # so our env overrides must be applied AFTER that.
    import cognee

    os.environ["GRAPH_DATABASE_PROVIDER"] = "helix"
    os.environ["VECTOR_DB_PROVIDER"] = "helix"
    os.environ["GRAPH_DATABASE_URL"] = HELIX_URL
    os.environ["VECTOR_DB_URL"] = HELIX_URL
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    # Clear cached configs + engine factories so they re-read the env overrides.
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.vector.config import get_vectordb_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    get_graph_config.cache_clear()
    get_vectordb_config.cache_clear()
    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = (
        "Helix is a unified graph and vector database. "
        "Cognee builds knowledge graphs from text using an ECL pipeline."
    )
    await cognee.add(text)
    await cognee.cognify()

    from cognee.modules.search.types import SearchType

    results = await cognee.search(
        query_text="What is Helix?", query_type=SearchType.GRAPH_COMPLETION
    )
    assert results, "expected non-empty GRAPH_COMPLETION results"

    chunks = await cognee.search(query_text="unified database", query_type=SearchType.CHUNKS)
    assert chunks, "expected non-empty CHUNKS results"

    print("HelixDB e2e add -> cognify -> search succeeded.")


@pytest.mark.asyncio
@pytest.mark.skipif(not _helix_reachable(), reason="HelixDB gateway not reachable")
@pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")
async def test_helix_hybrid_end_to_end():
    await main()


if __name__ == "__main__":
    asyncio.run(main())
