"""
Test for cognify tool

These are integration tests that test the actual tool behavior.
Run with: pytest tests/tools/test_cognify.py -v
"""

import pytest
import mcp.types as types

from src import server


@pytest.mark.asyncio
async def test_cognify():
    """Test cognify tool - launches background task to process data"""
    result = await server.cognify(data="Test data for cognify")

    assert len(result) == 1
    assert isinstance(result[0], types.TextContent)
    assert "Background process" in result[0].text or "launched" in result[0].text
