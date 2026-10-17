from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.klaviyo import _get_headers, klaviyo_source


def test_klaviyo_source_requires_token(monkeypatch):
    monkeypatch.delenv("KLAVIYO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Klaviyo API key must be provided"):
        klaviyo_source(api_key=None)


def test_klaviyo_get_headers():
    headers = _get_headers("my_token")
    assert headers["Authorization"] == "Klaviyo-API-Key my_token"
    assert "revision" in headers
    assert headers["accept"] == "application/json"


def test_klaviyo_source_tags():
    from cognee.tasks.ingestion.dlt_utils import document_source_tag

    source = klaviyo_source(api_key="my_test_token")
    assert document_source_tag(source) == "klaviyo"
