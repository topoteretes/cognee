import pytest
import cognee
from cognee.modules.data.exceptions import SearchOnEmptyGraphError


@pytest.mark.asyncio
async def test_empty_search_raises_SearchOnEmptyGraphError_on_empty_graph():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Sample input")
    with pytest.raises(SearchOnEmptyGraphError):
        await cognee.search("Sample query")


async def test_empty_search_doesnt_raise_SearchOnEmptyGraphError():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add("Sample input")
    await cognee.cognify()
    try:
        await cognee.search("Sample query")
    except SearchOnEmptyGraphError:
        pytest.fail("Should not raise SearchOnEmptyGraphError when data was added and cognified")
