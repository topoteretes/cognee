from unittest.mock import MagicMock, patch

import httpx
import pytest

from cognee.tasks.ingestion.connectors.airtable import _get_headers, airtable_source


def test_airtable_source_requires_token(monkeypatch):
    monkeypatch.delenv("AIRTABLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Airtable token must be provided"):
        airtable_source(token=None)


def test_airtable_get_headers():
    headers = _get_headers("my_token")
    assert headers["Authorization"] == "Bearer my_token"
    assert headers["Content-Type"] == "application/json"


def test_airtable_source_tags():
    from cognee.tasks.ingestion.dlt_utils import document_source_tag

    source = airtable_source(token="my_test_token")
    assert document_source_tag(source) == "airtable"
