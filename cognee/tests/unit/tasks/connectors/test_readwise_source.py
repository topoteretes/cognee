import pytest
from unittest.mock import MagicMock, patch

from cognee.tasks.ingestion.connectors.readwise import readwise_source, _get_headers

def test_readwise_source_requires_token(monkeypatch):
    monkeypatch.delenv("READWISE_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Readwise access token must be provided"):
        readwise_source(token=None)

def test_readwise_get_headers():
    headers = _get_headers("my_token")
    assert headers["Authorization"] == "Token my_token"
    assert headers["Content-Type"] == "application/json"
