import pytest


@pytest.fixture(autouse=True)
def allow_http_requests(monkeypatch):
    """These tests fetch live pages; ``ALLOW_HTTP_REQUESTS`` is off by default."""
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
