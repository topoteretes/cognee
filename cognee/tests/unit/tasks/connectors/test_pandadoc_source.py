"""Tests for PandaDoc data-source connector: authentication, pricing tables, incremental sync, and tombstones."""

from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.pandadoc import (
    STATUS_NAME_TO_CODE,
    PandaDocAuthError,
    PandaDocClient,
    PandaDocRateLimitedError,
    PandaDocResourceNotFoundError,
    PandaDocSettings,
    _iter_pandadoc_rows,
    _render_document_markdown,
    pandadoc_source,
)


def test_client_validation():
    with pytest.raises(ValueError, match="api_key must not be empty"):
        PandaDocClient(api_key="")

    with pytest.raises(ValueError, match="api_key must not be empty"):
        PandaDocClient(api_key="   ")


def test_client_key_masking():
    client = PandaDocClient(api_key="secret_pandadoc_key_12345")
    rep = repr(client)
    assert "secret_pandadoc_key_12345" not in rep
    assert "***" in rep
    assert "https://api.pandadoc.com/public/v1" in rep


def test_client_headers():
    client = PandaDocClient(api_key="my_key_abc")
    headers = client._headers()
    assert headers["Authorization"] == "API-Key my_key_abc"
    assert headers["Accept"] == "application/json"
    assert "cognee-pandadoc-connector" in headers["User-Agent"]


def test_client_auth_error_handling():
    import urllib.error

    client = PandaDocClient(api_key="invalid_key")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.pandadoc.com/public/v1/documents",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(PandaDocAuthError, match="authentication failed \\(HTTP 401\\)"),
    ):
        client.list_documents()


def test_client_resource_not_found():
    import urllib.error

    client = PandaDocClient(api_key="key")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.pandadoc.com/public/v1/documents/missing/details",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(PandaDocResourceNotFoundError, match="resource not found \\(HTTP 404\\)"),
    ):
        client.get_document_details("missing")


def test_client_rate_limited():
    import urllib.error

    client = PandaDocClient(api_key="key", max_retries=1)

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.pandadoc.com/public/v1/documents",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "4"},
            fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(PandaDocRateLimitedError) as exc_info:
            client.list_documents()
        assert exc_info.value.retry_after == 4


def test_status_name_mapping():
    assert STATUS_NAME_TO_CODE["draft"] == 0
    assert STATUS_NAME_TO_CODE["sent"] == 1
    assert STATUS_NAME_TO_CODE["completed"] == 2
    assert STATUS_NAME_TO_CODE["document.completed"] == 2
    assert STATUS_NAME_TO_CODE["voided"] == 11


def test_render_document_markdown():
    details = {
        "id": "doc_acme_2026",
        "name": "Acme Annual Cloud License Agreement",
        "status": "document.completed",
        "date_created": "2026-10-01T10:00:00Z",
        "date_modified": "2026-10-05T14:30:00Z",
        "date_sent": "2026-10-02T09:00:00Z",
        "date_completed": "2026-10-05T14:30:00Z",
        "grand_total": {"amount": "50000.00", "currency": "USD"},
        "tags": ["enterprise", "q4-deals"],
        "recipients": [
            {
                "first_name": "Alice",
                "last_name": "Smith",
                "email": "alice@acme.com",
                "role": "Signer",
                "has_completed": True,
            },
            {
                "first_name": "Bob",
                "last_name": "Jones",
                "email": "bob@acme.com",
                "role": "CC",
                "has_completed": False,
            },
        ],
        "pricing": {
            "tables": [
                {
                    "name": "Software Subscriptions",
                    "items": [
                        {
                            "name": "Core Platform License",
                            "description": "50 Seats @ $800/yr",
                            "qty": 50,
                            "price": "800.00",
                            "discount": "10%",
                            "subtotal": "36000.00",
                        },
                        {
                            "name": "24/7 Dedicated Support",
                            "description": "Enterprise SLA",
                            "qty": 1,
                            "price": "14000.00",
                            "discount": "0%",
                            "subtotal": "14000.00",
                        },
                    ],
                }
            ]
        },
        "fields": [{"name": "Contract Term", "value": "12 Months"}],
        "tokens": [{"name": "Account.Region", "value": "North America"}],
        "metadata": {"deal_crm_id": "DEAL-8891"},
    }

    md = _render_document_markdown(details)

    assert "# PandaDoc: Acme Annual Cloud License Agreement" in md
    assert "- **Document ID**: `doc_acme_2026`" in md
    assert "- **Status**: `document.completed`" in md
    assert "- **Grand Total**: `USD 50000.00`" in md
    assert "- **Alice Smith** (`alice@acme.com`) - Role: `Signer` | Signed: `Yes (Signed)`" in md
    assert "- **Bob Jones** (`bob@acme.com`) - Role: `CC` | Signed: `Pending`" in md
    assert "| Core Platform License | 50 Seats @ $800/yr | 50 | 800.00 | 10% | 36000.00 |" in md
    assert "| 24/7 Dedicated Support | Enterprise SLA | 1 | 14000.00 | 0% | 14000.00 |" in md
    assert "- **Contract Term**: 12 Months" in md
    assert "- **[Account.Region]**: North America" in md
    assert "- **`deal_crm_id`**: DEAL-8891" in md


def test_iter_pandadoc_rows_ingest_and_cursor():
    client = MagicMock(spec=PandaDocClient)
    client.list_documents.side_effect = [
        # Active documents listing
        {
            "results": [
                {
                    "id": "doc_1",
                    "name": "Service Quote #101",
                    "status": "document.completed",
                    "date_modified": "2026-10-15T12:00:00Z",
                }
            ],
            "total": 1,
        },
        # Empty page 2
        {"results": [], "total": 1},
        # Deletions listing (empty)
        {"results": [], "total": 0},
    ]
    client.get_document_details.return_value = {
        "id": "doc_1",
        "name": "Service Quote #101",
        "status": "document.completed",
        "date_created": "2026-10-10T09:00:00Z",
        "date_modified": "2026-10-15T12:00:00Z",
        "grand_total": {"amount": "12000.00", "currency": "USD"},
        "recipients": [],
        "pricing": {"tables": []},
    }

    state = {"cursor": None, "seen_ids": []}
    stats = {}

    rows = list(
        _iter_pandadoc_rows(
            client=client,
            status_codes=[2],
            tag=None,
            folder_uuid=None,
            state=state,
            stats=stats,
        )
    )

    assert len(rows) == 1
    doc = rows[0]
    assert doc["id"] == "pandadoc:doc:doc_1"
    assert doc["_deleted"] is False
    assert "Service Quote #101" in doc["title"]
    assert "USD 12000.00" in doc["content"]
    assert doc["cognee_node_set"] == ["pandadoc:documents"]

    # Cursor check
    assert state["cursor"] == "2026-10-15T12:00:00Z"
    assert "doc_1" in state["seen_ids"]


def test_iter_pandadoc_rows_deleted_tombstone():
    client = MagicMock(spec=PandaDocClient)
    client.list_documents.side_effect = [
        # Active documents listing (empty)
        {"results": [], "total": 0},
        # Deletions listing with 1 deleted document
        {
            "results": [
                {
                    "id": "doc_deleted_99",
                    "name": "Expired Proposal",
                    "status": "document.deleted",
                }
            ],
            "total": 1,
        },
        # Empty next deletion page
        {"results": [], "total": 1},
    ]

    state = {"cursor": "2026-10-01T00:00:00Z", "seen_ids": []}
    stats = {}

    rows = list(
        _iter_pandadoc_rows(
            client=client,
            status_codes=None,
            tag=None,
            folder_uuid=None,
            state=state,
            stats=stats,
        )
    )

    assert len(rows) == 1
    tombstone = rows[0]
    assert tombstone["id"] == "pandadoc:doc:doc_deleted_99"
    assert tombstone["_deleted"] is True
    assert stats["deleted"] == 1


def test_pandadoc_settings_env_loading(monkeypatch):
    monkeypatch.setenv("PANDADOC_API_KEY", "env_secret_key_88")
    monkeypatch.setenv("PANDADOC_BASE_URL", "https://api-sandbox.pandadoc.com/public/v1")
    monkeypatch.setenv("PANDADOC_STATUSES", '["sent", "completed"]')

    settings = PandaDocSettings()
    assert settings.pandadoc_api_key == "env_secret_key_88"
    assert settings.pandadoc_base_url == "https://api-sandbox.pandadoc.com/public/v1"
    assert settings.pandadoc_statuses == ["sent", "completed"]


def test_pandadoc_source_full_crawl():
    client = MagicMock(spec=PandaDocClient)
    client.list_documents.side_effect = [
        # Active documents
        {
            "results": [
                {
                    "id": "doc_final",
                    "name": "Master Services Agreement",
                    "status": "document.completed",
                    "date_modified": "2026-10-18T16:00:00Z",
                }
            ],
            "total": 1,
        },
        {"results": [], "total": 1},
        # Deletions
        {"results": [], "total": 0},
    ]
    client.get_document_details.return_value = {
        "id": "doc_final",
        "name": "Master Services Agreement",
        "status": "document.completed",
        "date_created": "2026-10-10T00:00:00Z",
        "date_modified": "2026-10-18T16:00:00Z",
        "grand_total": {"amount": "100000.00", "currency": "USD"},
        "recipients": [],
        "pricing": {"tables": []},
    }

    source = pandadoc_source(
        client=client,
        statuses=["completed"],
    )

    items = list(source)
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "pandadoc:doc:doc_final"
    assert item["title"] == "PandaDoc: Master Services Agreement"
    assert item["_deleted"] is False
    assert "USD 100000.00" in item["content"]
