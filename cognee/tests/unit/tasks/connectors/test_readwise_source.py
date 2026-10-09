import pytest
from unittest.mock import MagicMock, patch

from cognee.tasks.ingestion.connectors.readwise import readwise_source, _get_headers

def test_readwise_source_requires_token(monkeypatch):
    monkeypatch.delenv("READWISE_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Readwise access token must be provided"):
        readwise_source(token=None)
