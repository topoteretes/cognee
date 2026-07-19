import sys
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.cypher_search_retriever import CypherSearchRetriever
from cognee.modules.retrieval.natural_language_retriever import NaturalLanguageRetriever

POSTGRES_ADAPTER_MODULES = [
    "cognee.infrastructure.databases.graph.postgres.adapter",
    "cognee.infrastructure.databases.hybrid.postgres.adapter",
]


@pytest.fixture
def missing_postgres_driver(monkeypatch):
    """Force the Postgres adapter imports to fail, as on installs without the postgres extra."""
    for module_name in POSTGRES_ADAPTER_MODULES:
        monkeypatch.setitem(sys.modules, module_name, None)


@pytest.mark.asyncio
async def test_cypher_search_executes_without_postgres_driver(missing_postgres_driver):
    """The query must still run when the optional Postgres driver is not installed."""
    mock_engine = AsyncMock()
    mock_engine.is_empty = AsyncMock(return_value=False)
    mock_engine.query = AsyncMock(return_value=[(10,)])

    retriever = CypherSearchRetriever()

    with patch(
        "cognee.modules.retrieval.cypher_search_retriever.get_graph_engine",
        return_value=mock_engine,
    ):
        result = await retriever.get_retrieved_objects("MATCH (n) RETURN count(n)")

    assert result == [(10,)]
    mock_engine.query.assert_awaited_once_with("MATCH (n) RETURN count(n)")


@pytest.mark.asyncio
async def test_natural_language_search_executes_without_postgres_driver(missing_postgres_driver):
    """The backend check must not require the optional Postgres driver."""
    mock_engine = AsyncMock()
    mock_engine.is_empty = AsyncMock(return_value=False)

    retriever = NaturalLanguageRetriever()

    with (
        patch(
            "cognee.modules.retrieval.natural_language_retriever.get_graph_engine",
            return_value=mock_engine,
        ),
        patch.object(
            NaturalLanguageRetriever,
            "_execute_cypher_query",
            AsyncMock(return_value=[("node_count", 10)]),
        ),
    ):
        result = await retriever.get_retrieved_objects("How many nodes are there?")

    assert result == [("node_count", 10)]
