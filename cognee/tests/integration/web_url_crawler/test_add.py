import pytest
import cognee


@pytest.mark.asyncio
async def test_add_fails_when_preferred_loader_not_specified():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    with pytest.raises(ValueError):
        await cognee.add(
            "https://en.wikipedia.org/wiki/Large_language_model",
            incremental_loading=False,  # TODO: incremental loading bypasses regular data ingestion, which breaks. Will fix
        )


@pytest.mark.asyncio
async def test_add_succesfully_adds_url_when_preferred_loader_specified():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title", "attr": "text"},
        "headings": {"selector": "h1, h2, h3", "attr": "text", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "attr": "text", "all": True},
    }

    loaders_config = {
        "web_url_loader": {
            "soup_config": {
                "max_depth": 1,
                "follow_links": False,
                "extraction_rules": extraction_rules,
            }
        }
    }

    try:
        await cognee.add(
            "https://en.wikipedia.org/wiki/Large_language_model",
            preferred_loaders=["web_url_loader"],
            incremental_loading=False,  # TODO: incremental loading bypasses regular data ingestion, which breaks. Will fix
            loaders_config=loaders_config,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")
