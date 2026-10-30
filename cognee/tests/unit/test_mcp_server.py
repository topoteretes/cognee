"""Unit tests for Cognee FastMCP server integration."""

import pytest
from unittest.mock import AsyncMock

from cognee.api.mcp import (
    server,
    SearchTypeEnum,
    AddMemoryResponse,
    SearchMemoryResponse,
    PruneMemoryResponse,
)
from cognee.api.mcp.adapter import CogneeAdapter


class MockCogneeAdapter(CogneeAdapter):
    """Mock implementation of CogneeAdapter for deterministic testing."""

    def __init__(self):
        super().__init__()
        self.memories = []
        self.add_memory_mock = AsyncMock(side_effect=self._default_add_memory)
        self.search_memory_mock = AsyncMock(side_effect=self._default_search_memory)
        self.prune_memory_mock = AsyncMock(side_effect=self._default_prune_memory)

    async def _default_add_memory(
        self,
        content: str,
        dataset_name: str = "default_dataset",
        auto_cognify: bool = True
    ) -> AddMemoryResponse:
        self.memories.append({"content": content, "dataset": dataset_name, "cognified": auto_cognify})
        return AddMemoryResponse(
            success=True,
            dataset_name=dataset_name,
            message=f"Successfully added content to '{dataset_name}'.",
            cognified=auto_cognify,
            error=None
        )

    async def _default_search_memory(
        self,
        query: str,
        search_type: SearchTypeEnum = SearchTypeEnum.INSIGHTS,
        dataset_name: str = None
    ) -> SearchMemoryResponse:
        filtered = [m for m in self.memories if not dataset_name or m["dataset"] == dataset_name]
        return SearchMemoryResponse(
            success=True,
            query=query,
            search_type=search_type.value,
            dataset_name=dataset_name,
            total_results=len(filtered),
            results=filtered,
            error=None
        )

    async def _default_prune_memory(self, prune_all: bool = False) -> PruneMemoryResponse:
        self.memories.clear()
        return PruneMemoryResponse(
            success=True,
            message="Prune completed successfully.",
            prune_all=prune_all
        )

    async def add_memory(self, *args, **kwargs):
        return await self.add_memory_mock(*args, **kwargs)

    async def search_memory(self, *args, **kwargs):
        return await self.search_memory_mock(*args, **kwargs)

    async def prune_memory(self, *args, **kwargs):
        return await self.prune_memory_mock(*args, **kwargs)


@pytest.fixture
def mock_adapter():
    """Provides a fresh MockCogneeAdapter and sets it into the server."""
    adapter = MockCogneeAdapter()
    server.set_adapter(adapter)
    yield adapter
    server.set_adapter(server.default_adapter)


@pytest.mark.asyncio
async def test_add_memory_success(mock_adapter):
    """Test successful memory ingestion into knowledge graph."""
    res = await server.cognee_add_memory(
        content="Antigravity builds autonomous AI systems.",
        dataset_name="research_docs",
        auto_cognify=True
    )

    assert isinstance(res, AddMemoryResponse)
    assert res.success is True
    assert res.dataset_name == "research_docs"
    assert res.cognified is True
    assert res.error is None
    assert len(mock_adapter.memories) == 1
    assert mock_adapter.memories[0]["content"] == "Antigravity builds autonomous AI systems."


@pytest.mark.asyncio
async def test_add_memory_empty_content(mock_adapter):
    """Test validation when content is empty or blank."""
    res1 = await server.cognee_add_memory(content="")
    assert res1.success is False
    assert "required" in res1.error.lower()

    res2 = await server.cognee_add_memory(content="    ")
    assert res2.success is False
    assert "blank" in res2.error.lower()
    assert len(mock_adapter.memories) == 0


@pytest.mark.asyncio
async def test_add_memory_exception_handling(mock_adapter):
    """Test error handling when downstream ingestion fails."""
    mock_adapter.add_memory_mock = AsyncMock(
        return_value=AddMemoryResponse(
            success=False,
            dataset_name="default_dataset",
            message="Ingestion failed.",
            cognified=False,
            error="Connection to Cognee cluster timed out."
        )
    )

    res = await server.cognee_add_memory(content="Important Note")
    assert res.success is False
    assert "timed out" in res.error


@pytest.mark.asyncio
async def test_search_memory_success(mock_adapter):
    """Test retrieving memory with search."""
    await server.cognee_add_memory(content="Knowledge graph architecture details", dataset_name="tech_spec")
    await server.cognee_add_memory(content="Unrelated meeting notes", dataset_name="admin")

    res = await server.cognee_search_memory(
        query="graph architecture",
        search_type="INSIGHTS",
        dataset_name="tech_spec"
    )

    assert isinstance(res, SearchMemoryResponse)
    assert res.success is True
    assert res.query == "graph architecture"
    assert res.search_type == "INSIGHTS"
    assert res.dataset_name == "tech_spec"
    assert res.total_results == 1
    assert res.results[0]["dataset"] == "tech_spec"


@pytest.mark.asyncio
async def test_search_memory_case_insensitivity(mock_adapter):
    """Test case-insensitive search type parsing."""
    res = await server.cognee_search_memory(query="test", search_type="chunks")
    assert res.success is True
    assert res.search_type == "CHUNKS"

    res_fallback = await server.cognee_search_memory(query="test", search_type="UNKNOWN_MODE")
    assert res_fallback.success is True
    assert res_fallback.search_type == "INSIGHTS"


@pytest.mark.asyncio
async def test_search_memory_empty_query(mock_adapter):
    """Test searching with an empty query triggers validation."""
    res = await server.cognee_search_memory(query="")
    assert res.success is False
    assert "query" in res.error.lower()
    assert res.total_results == 0


@pytest.mark.asyncio
async def test_search_memory_exception_handling(mock_adapter):
    """Test graceful error response when search fails."""
    mock_adapter.search_memory_mock = AsyncMock(
        return_value=SearchMemoryResponse(
            success=False,
            query="fail test",
            search_type="INSIGHTS",
            total_results=0,
            results=[],
            error="Vector database unavailable"
        )
    )

    res = await server.cognee_search_memory(query="fail test")
    assert res.success is False
    assert "Vector database unavailable" in res.error


@pytest.mark.asyncio
async def test_prune_memory(mock_adapter):
    """Test pruning data and system."""
    await server.cognee_add_memory(content="To be deleted")
    assert len(mock_adapter.memories) == 1

    res = await server.cognee_prune_memory(prune_all=False)
    assert res.success is True
    assert len(mock_adapter.memories) == 0


@pytest.mark.asyncio
async def test_mcp_call_tool_dispatch(mock_adapter):
    """Test direct MCP protocol tool execution via mcp.call_tool()."""
    add_result = await server.mcp.call_tool(
        "cognee_add_memory",
        {"content": "MCP Standard Protocol Memory Entry", "dataset_name": "mcp_dataset"}
    )
    assert add_result.is_error is False

    search_result = await server.mcp.call_tool(
        "cognee_search_memory",
        {"query": "Protocol Memory", "search_type": "INSIGHTS", "dataset_name": "mcp_dataset"}
    )
    assert search_result.is_error is False


@pytest.mark.asyncio
async def test_mcp_tools_registered():
    """Verify that all Cognee tools are properly registered with FastMCP."""
    tools = await server.mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "cognee_add_memory" in tool_names
    assert "cognee_search_memory" in tool_names
    assert "cognee_prune_memory" in tool_names
