from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.calendly import _get_headers, calendly_source


def test_calendly_source_requires_token(monkeypatch):
    monkeypatch.delenv("CALENDLY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Calendly API key must be provided"):
        calendly_source(token=None)


def test_calendly_get_headers():
    headers = _get_headers("my_token")
    assert headers["Authorization"] == "Bearer my_token"
    assert headers["Content-Type"] == "application/json"


def test_calendly_source_tags():
    from cognee.tasks.ingestion.dlt_utils import document_source_tag

    source = calendly_source(token="my_test_token")
    assert document_source_tag(source) == "calendly"
