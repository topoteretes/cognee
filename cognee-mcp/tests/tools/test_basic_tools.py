"""
Tests for basic MCP tools: prune, cognify_status

These are integration tests that test the actual tool behavior.
Run with: pytest tests/tools/test_basic_tools.py -v
"""

import pytest
import mcp.types as types

from src import server


@pytest.mark.asyncio
async def test_prune():
    """Test prune tool - removes all data from knowledge graph"""
    result = await server.prune()

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert "Pruned" in result[0].text or "not available" in result[0].text


@pytest.mark.asyncio
async def test_cognify_status():
    """Test cognify_status tool - gets status of cognify pipeline"""
    result = await server.cognify_status()

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert len(result[0].text) > 0
