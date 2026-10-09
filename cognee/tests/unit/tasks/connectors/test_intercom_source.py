import pytest
from unittest.mock import MagicMock, patch

from cognee.tasks.ingestion.connectors.intercom import intercom_source, _get_headers

def test_intercom_source_requires_token(monkeypatch):
    monkeypatch.delenv("INTERCOM_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Intercom access token must be provided"):
        intercom_source(token=None)

def test_intercom_get_headers():
    headers = _get_headers("my_token")
    assert headers["Authorization"] == "Bearer my_token"
    assert headers["Accept"] == "application/json"
    assert headers["Intercom-Version"] == "2.11"
