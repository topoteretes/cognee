"""Unit tests for the Teable no-code database data-source connector.

Tests pagination, content-hash change detection, cross-table relational links,
deletion tombstones, trash reconciliation, rate limits, and token hygiene.
All tests use in-memory mock clients with zero external network requests.
"""

from typing import Any

import httpx
import pytest

from cognee.tasks.ingestion.connectors.teable import (
    TeableAPIError,
    TeableAuthError,
    TeableClient,
    TeableRateLimitError,
    TeableSettings,
    compute_content_hash,
    format_cell_value,
    iter_teable_records,
    mask_token,
    render_teable_record,
    teable_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR


class MockTeableTransport(httpx.BaseTransport):
    """Mock HTTP transport returning controlled Teable API responses."""

    def __init__(self, handler):
        self.handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.handler(request)


def test_token_masking():
    """Verify that bearer tokens are masked across string representations."""
    assert mask_token(None) == "<not-set>"
    assert mask_token("short") == "***"
    assert mask_token("teable_pat_998877665544332211") == "teab...2211"

    client = TeableClient(api_key="teable_pat_998877665544332211")
    repr_str = repr(client)
    assert "teable_pat_998877665544332211" not in repr_str
    assert "teab...2211" in repr_str


def test_compute_content_hash():
    """Verify deterministic content hashing for change detection."""
    content_a = "# Table: Record 1\nProperty: Value A"
    content_b = "# Table: Record 1\nProperty: Value B"
    assert compute_content_hash(content_a) == compute_content_hash(content_a)
    assert compute_content_hash(content_a) != compute_content_hash(content_b)


def test_format_cell_value():
    """Verify formatting of various cell types including single and multiple link objects."""
    assert format_cell_value(None) == ("N/A", [])
    assert format_cell_value(True) == ("Yes", [])
    assert format_cell_value(False) == ("No", [])
    assert format_cell_value(42) == ("42", [])
    assert format_cell_value("Plain Text") == ("Plain Text", [])

    # Single link object
    single_link = {"id": "recComp1", "title": "Acme Corp"}
    val_str, links = format_cell_value(single_link)
    assert val_str == "Acme Corp"
    assert links == [{"id": "recComp1", "title": "Acme Corp"}]

    # Multiple link objects
    multi_links = [
        {"id": "recC1", "title": "Contact A"},
        {"id": "recC2", "title": "Contact B"},
    ]
    val_str, links = format_cell_value(multi_links)
    assert val_str == "Contact A, Contact B"
    assert len(links) == 2
    assert links[0]["id"] == "recC1"


def test_render_teable_record_basic():
    """Verify basic record rendering into structured markdown."""
    raw_record = {
        "id": "rec_001",
        "name": "Acme Corporation",
        "createdTime": "2026-10-18T10:00:00Z",
        "lastModifiedTime": "2026-10-18T10:30:00Z",
        "fields": {
            "Industry": "Software",
            "Employees": 250,
            "IsEnterprise": True,
        },
    }

    doc = render_teable_record(raw_record, table_id="tbl_companies", table_name="Companies")
    assert doc["id"] == "teable:tbl_companies:rec_001"
    assert doc["title"] == "Companies: Acme Corporation"
    assert doc["table_id"] == "tbl_companies"
    assert doc["table_name"] == "Companies"
    assert "Acme Corporation" in doc["content"]
    assert "**Industry**: Software" in doc["content"]
    assert "**Employees**: 250" in doc["content"]
    assert "**IsEnterprise**: Yes" in doc["content"]
    assert doc["content_hash"] is not None


def test_render_teable_record_with_links():
    """Verify rendering of cross-table link fields into semantic relations."""
    raw_record = {
        "id": "rec_contact_99",
        "name": "Sarah Connor",
        "fields": {
            "Company": [{"id": "rec_comp_123", "title": "Cyberdyne Systems"}],
            "Role": "Lead Engineer",
        },
    }

    doc = render_teable_record(raw_record, table_id="tbl_contacts", table_name="Contacts")
    assert doc["id"] == "teable:tbl_contacts:rec_contact_99"
    assert "[Cyberdyne Systems](teable:rec_comp_123)" in doc["content"]
    assert "**Role**: Lead Engineer" in doc["content"]


def test_render_teable_record_tombstone():
    """Verify deleted record flag produces tombstone."""
    raw_record = {
        "id": "rec_deleted_001",
        "is_deleted": True,
    }
    tombstone = render_teable_record(raw_record, table_id="tbl_test")
    assert tombstone == {"id": "teable:tbl_test:rec_deleted_001", "_deleted": True}


def test_iter_teable_records_content_hash_deduplication():
    """Verify that unchanged records are skipped on subsequent runs while changed records are yielded."""
    first_payload = {
        "records": [
            {
                "id": "rec_1",
                "name": "Alpha Project",
                "fields": {"Status": "In Progress"},
            },
            {
                "id": "rec_2",
                "name": "Beta Project",
                "fields": {"Status": "Planning"},
            },
        ],
        "extra": {"nextCursor": None},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=first_payload)

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}

    # Run 1: both records are new and must be yielded
    records_1 = list(iter_teable_records(client, table_ids=["tbl_proj"], state=state, stats=stats))
    assert len(records_1) == 2
    assert stats["synced_records"] == 2
    assert stats.get("skipped_unchanged", 0) == 0

    # Run 2: same content without changes -> both skipped
    stats.clear()
    records_2 = list(iter_teable_records(client, table_ids=["tbl_proj"], state=state, stats=stats))
    assert len(records_2) == 0
    assert stats["skipped_unchanged"] == 2

    # Run 3: update rec_1 status -> rec_1 yielded, rec_2 skipped
    first_payload["records"][0]["fields"]["Status"] = "Completed"
    stats.clear()
    records_3 = list(iter_teable_records(client, table_ids=["tbl_proj"], state=state, stats=stats))
    assert len(records_3) == 1
    assert records_3[0]["id"] == "teable:tbl_proj:rec_1"
    assert "**Status**: Completed" in records_3[0]["content"]
    assert stats["synced_records"] == 1
    assert stats["skipped_unchanged"] == 1


def test_iter_teable_records_deletion_tombstones():
    """Verify that records missing from subsequent complete table listings emit tombstones."""
    current_records = [
        {"id": "rec_stay", "name": "Active Record", "fields": {"Val": 1}},
        {"id": "rec_remove", "name": "To be deleted", "fields": {"Val": 2}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"records": current_records, "extra": {"nextCursor": None}})

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}

    # Run 1: ingest both
    list(iter_teable_records(client, table_ids=["tbl_demo"], state=state, stats=stats))

    # Run 2: rec_remove is gone from listing
    current_records = [current_records[0]]
    stats.clear()
    records_2 = list(iter_teable_records(client, table_ids=["tbl_demo"], state=state, stats=stats))

    assert len(records_2) == 1
    assert records_2[0] == {"id": "teable:tbl_demo:rec_remove", "_deleted": True}
    assert stats["deleted_tombstones"] == 1


def test_iter_teable_records_trash_reconciliation():
    """Verify that records found in Teable trash items are confirmed as tombstones."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/trash/items":
            return httpx.Response(
                200,
                json={
                    "trashItems": [
                        {"id": "trash_1", "resourceId": "rec_trashed", "deletedTime": "2026-10-18"}
                    ]
                },
            )
        return httpx.Response(200, json={"records": [], "extra": {"nextCursor": None}})

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    state: dict[str, Any] = {
        "known_records": {"tbl_trash": ["rec_trashed"]},
        "content_hashes": {"rec_trashed": "hash123"},
    }
    stats: dict[str, int] = {}
    records = list(iter_teable_records(client, table_ids=["tbl_trash"], state=state, stats=stats))

    assert len(records) == 1
    assert records[0] == {"id": "teable:tbl_trash:rec_trashed", "_deleted": True}
    assert "rec_trashed" not in state["content_hashes"]


def test_iter_teable_records_pagination():
    """Verify pagination across multiple pages with nextCursor."""
    page = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page
        page += 1
        if page == 1:
            return httpx.Response(
                200,
                json={
                    "records": [{"id": "rec_p1", "name": "Page 1 Item", "fields": {}}],
                    "extra": {"nextCursor": "cursor_page_2"},
                },
            )
        return httpx.Response(
            200,
            json={
                "records": [{"id": "rec_p2", "name": "Page 2 Item", "fields": {}}],
                "extra": {"nextCursor": None},
            },
        )

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}
    records = list(iter_teable_records(client, table_ids=["tbl_pages"], state=state, stats=stats))

    assert len(records) == 2
    assert records[0]["id"] == "teable:tbl_pages:rec_p1"
    assert records[1]["id"] == "teable:tbl_pages:rec_p2"


def test_rate_limit_graceful_abort():
    """Verify 429 rate limit triggers graceful abort without raising or discarding state."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too Many Requests", "code": 429})

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}
    records = list(iter_teable_records(client, table_ids=["tbl_rl"], state=state, stats=stats))

    assert records == []
    assert stats["rate_limited"] == 1


def test_teable_source_builder_attributes():
    """Verify teable_source builder attaches correct metadata and attributes."""
    client = TeableClient(api_key="fake_pat")
    resource = teable_source(
        service=client,
        table_ids=["tbl_crm_1"],
        resource_name="teable_test_scope",
    )

    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "teable"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "teable_test_scope"
    assert hasattr(resource, "cognee_sync_stats")


def test_teable_client_auth_error():
    """Verify 401 raises TeableAuthError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Invalid token", "code": "UNAUTHORIZED"})

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="invalid_pat", http_client=mock_client)

    with pytest.raises(TeableAuthError) as exc_info:
        client.get_table_records(table_id="tbl_1")
    assert "authentication failed" in str(exc_info.value).lower()


def test_teable_client_general_error():
    """Verify general API error parsing and error code extraction."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Table not found", "code": "TABLE_NOT_FOUND"})

    mock_client = httpx.Client(transport=MockTeableTransport(handler))
    client = TeableClient(api_key="test_pat", http_client=mock_client)

    with pytest.raises(TeableAPIError) as exc_info:
        client.get_table_records(table_id="tbl_nonexistent")
    assert "Table not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404
    assert exc_info.value.error_code == "TABLE_NOT_FOUND"
