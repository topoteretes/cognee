from sys import exc_info
import pytest
import cognee
from cognee.modules.ingestion.exceptions.exceptions import IngestionError


@pytest.mark.asyncio
async def test_add_fails_when_web_url_fetcher_config_not_specified():
    from cognee.shared.logging_utils import setup_logging, ERROR

    setup_logging(log_level=ERROR)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    with pytest.raises(IngestionError) as excinfo:
        await cognee.add(
            "https://en.wikipedia.org/wiki/Large_language_model",
            incremental_loading=False,
        )
    assert excinfo.value.message.startswith(
        "web_url_fetcher configuration must be a valid dictionary"
    )


@pytest.mark.asyncio
async def test_add_succesfully_adds_url_when_fetcher_config_specified():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    fetchers_config = {
        "web_url_fetcher": {
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
            incremental_loading=False,
            fetchers_config=fetchers_config,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")


@pytest.mark.asyncio
async def test_add_with_incremental_loading_works():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    fetchers_config = {
        "web_url_fetcher": {
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
            incremental_loading=True,
            fetchers_config=fetchers_config,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")


@pytest.mark.asyncio
async def test_add_without_incremental_loading_works():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    extraction_rules = {
        "title": {"selector": "title"},
        "headings": {"selector": "h1, h2, h3", "all": True},
        "links": {"selector": "a", "attr": "href", "all": True},
        "paragraphs": {"selector": "p", "all": True},
    }

    fetchers_config = {
        "web_url_fetcher": {
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
            incremental_loading=False,
            fetchers_config=fetchers_config,
        )
    except Exception as e:
        pytest.fail(f"Failed to add url: {e}")
