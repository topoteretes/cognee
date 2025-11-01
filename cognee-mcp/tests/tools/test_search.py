"""
Test for search tool

These are integration tests that test the actual tool behavior.
Run with: pytest tests/tools/test_search.py -v
"""

import cognee
from cognee.infrastructure.databases.exceptions import DatabaseNotCreatedError
import pytest
import mcp.types as types
from cognee import SearchType

from src import server


@pytest.mark.asyncio
async def test_search_handles_database_not_ready():
    """Test search tool handles database not ready scenario gracefully"""
    with pytest.raises(DatabaseNotCreatedError):
        await server.search(search_query="test query", search_type="GRAPH_COMPLETION")


@pytest.mark.asyncio
async def test_search_graph_completion():
    """Test search with GRAPH_COMPLETION type"""
    await cognee.add("Artificial intelligence and machine learning are transforming technology.")
    await cognee.cognify()

    result = await server.search(
        search_query="What is AI?", search_type=SearchType.GRAPH_COMPLETION.value
    )

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text is not None


@pytest.mark.asyncio
async def test_search_rag_completion():
    """Test search with RAG_COMPLETION type"""
    await cognee.add("Python is a programming language that emphasizes readability.")
    await cognee.cognify()

    result = await server.search(
        search_query="What is Python?", search_type=SearchType.RAG_COMPLETION.value
    )

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text is not None


@pytest.mark.asyncio
async def test_search_chunks():
    """Test search with CHUNKS type"""
    await cognee.add("JavaScript is the language of the web browser.")
    await cognee.cognify()

    result = await server.search(search_query="web browser", search_type=SearchType.CHUNKS.value)

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text is not None


@pytest.mark.asyncio
async def test_search_summaries():
    """Test search with SUMMARIES type"""
    await cognee.add("Database systems manage and store structured data efficiently.")
    await cognee.cognify()

    result = await server.search(search_query="database", search_type=SearchType.SUMMARIES.value)

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text is not None


@pytest.mark.asyncio
async def test_search_feeling_lucky():
    """Test search with FEELING_LUCKY type"""
    await cognee.add("Machine learning models learn patterns from data.")
    await cognee.cognify()

    result = await server.search(
        search_query="learning", search_type=SearchType.FEELING_LUCKY.value
    )

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert result[0].text is not None
