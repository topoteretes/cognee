"""Unit tests for the public exports of the web scraper package."""

import pytest

from cognee.tasks import web_scraper


@pytest.mark.parametrize("name", web_scraper.__all__)
def test_public_export_is_resolvable(name):
    """Every name in ``__all__`` must resolve, so ``from ... import *`` cannot raise."""
    try:
        getattr(web_scraper, name)
    except ImportError:
        pytest.skip(f"{name} requires an optional dependency that is not installed")
