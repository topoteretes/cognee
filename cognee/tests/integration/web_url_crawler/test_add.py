import pytest
import cognee


@pytest.mark.asyncio
async def test_add_fails_when_preferred_loader_not_specified():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    with pytest.raises:
        await cognee.add(
            "https://en.wikipedia.org/wiki/Large_language_model",
            preferred_loaders=["web_url_loader"],
        )


@pytest.mark.asyncio
async def test_add_succesfully_adds_url_when_preferred_loader_specified():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    try:
        await cognee.add(
            "https://en.wikipedia.org/wiki/Large_language_model",
            preferred_loaders=["web_url_loader"],
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")
