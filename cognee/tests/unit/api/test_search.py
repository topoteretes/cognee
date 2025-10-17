import pytest
import cognee


@pytest.mark.asyncio
async def test_empty_search_raises_SearchOnEmptyGraphError_on_empty_graph():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Sample input")
    result = await cognee.search("Sample query")
    assert result == []


@pytest.mark.asyncio
async def test_empty_search_doesnt_raise_SearchOnEmptyGraphError():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Sample input")
    await cognee.cognify()
    result = await cognee.search("Sample query")
    assert result != []
