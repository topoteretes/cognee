"""Neptune deletes exact (source, target, relationship) edges and keeps their endpoints.

improve(review_conflicts=True) and the incremental code graph sweep call
delete_edge_triples without a fallback. The base interface raises
UnsupportedProvenanceCapability, so on Neptune those flows failed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the optional AWS dependencies when they are not installed: the adapter
# imports botocore.config at module level and langchain_aws behind a guard, and
# the macOS/Windows unit legs run without the aws/neptune extras.
if "langchain_aws" not in sys.modules:
    sys.modules["langchain_aws"] = MagicMock()
try:
    import botocore
except ImportError:
    mock_botocore = MagicMock()
    sys.modules["botocore"] = mock_botocore
    sys.modules["botocore.config"] = mock_botocore.config

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
from cognee.infrastructure.databases.provenance import EdgeIdentity


@pytest.mark.asyncio
async def test_delete_edge_triples_deletes_only_the_matching_relationships():
    stub = SimpleNamespace(_GRAPH_NODE_LABEL="COGNEE_NODE", query=AsyncMock(return_value=[]))
    relationship = "odd`type MATCH (n) DETACH DELETE n"

    await NeptuneGraphDB.delete_edge_triples(
        stub,
        [EdgeIdentity("a", "b", "knows"), EdgeIdentity("c", "d", relationship)],
    )

    query, params = stub.query.await_args.args
    assert "id(source) = edge.source" in query
    assert "id(target) = edge.target" in query
    assert "type(r) = edge.relationship" in query
    assert "DELETE r" in query and "DETACH" not in query
    assert relationship not in query
    assert params == {
        "edges": [
            {"source": "a", "target": "b", "relationship": "knows"},
            {"source": "c", "target": "d", "relationship": relationship},
        ]
    }


@pytest.mark.asyncio
async def test_delete_edge_triples_with_no_edges_sends_no_query():
    stub = SimpleNamespace(_GRAPH_NODE_LABEL="COGNEE_NODE", query=AsyncMock())

    await NeptuneGraphDB.delete_edge_triples(stub, [])

    stub.query.assert_not_awaited()
