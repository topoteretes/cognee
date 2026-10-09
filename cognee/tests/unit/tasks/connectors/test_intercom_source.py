import pytest
from unittest.mock import MagicMock, patch

from cognee.tasks.ingestion.connectors.intercom import intercom_source, _get_headers

def test_intercom_source_requires_token(monkeypatch):
    monkeypatch.delenv("INTERCOM_ACCESS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Intercom access token must be provided"):
        intercom_source(token=None)
