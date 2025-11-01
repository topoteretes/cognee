"""
Tests for data management tools: list_data, delete

These are integration tests that test the actual tool behavior.
Run with: pytest tests/tools/test_data_tools.py -v
"""

import pytest
import mcp.types as types
import cognee

from src import server


@pytest.mark.asyncio
async def test_list_data_when_empty():
    """Test list_data tool - lists all datasets"""
    result = await server.list_data()

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert (
        "❌ Failed to list data: DatabaseNotCreatedError: The database has not been created yet"
        in result[0].text
    )


@pytest.mark.asyncio
async def test_list_data_when_cognified():
    """Test list_data tool - lists all datasets"""
    await cognee.add("Every node tells a story. Most of mine are unit tests.")
    await cognee.cognify()

    result = await server.list_data()

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert all(s in result[0].text for s in ["📂 Available Datasets", "Dataset ID: "])
