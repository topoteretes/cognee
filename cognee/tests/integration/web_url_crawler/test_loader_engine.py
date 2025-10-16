import pytest

from cognee.infrastructure.loaders import get_loader_engine
from cognee.infrastructure.loaders.external.web_url_loader import WebUrlLoader


def test_get_loader_returns_none_by_default_for_web_urls():
    loader_engine = get_loader_engine()
    urls = ["https://cognee.ai", "http://cognee.ai"]
    for url in urls:
        loader = loader_engine.get_loader(url)
        assert loader is None


def test_get_loader_returns_valid_loader_when_preferred_loaders_specified():
    loader_engine = get_loader_engine()
    urls = ["https://cognee.ai", "http://cognee.ai"]
    for url in urls:
        loader = loader_engine.get_loader(url, preferred_loaders=["web_url_loader"])
        assert isinstance(loader, WebUrlLoader)
