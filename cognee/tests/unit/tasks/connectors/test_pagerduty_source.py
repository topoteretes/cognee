"""Unit tests for the PagerDuty DLT connector: paging, cursors, resumption, rendering, and error handling."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors.pagerduty import (
    DEFAULT_PAGE_SIZE,
    PagerDutyAPIError,
    PagerDutyAuthError,
    PagerDutyClient,
    PagerDutyRateLimitedError,
    _default_resource_name,
    _int_header,
    _iter_rows,
    _retry_after,
    pagerduty_source,
    render_incident,
)

SAMPLE_TOKEN = "u+test_pagerduty_api_token_12345"


class FakeHTTPResponse:
    """Mock HTTP response for testing PagerDutyClient."""

    def __init__(self, status_code: int = 200, json_data: Any = None, headers: dict | None = None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = str(json_data or "")

    def json(self):
        return self._json


class FakeHTTPClient:
    """Mock HTTP client implementing request() method."""

    def __init__(self, responses: list[FakeHTTPResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            return FakeHTTPResponse(200, {})
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# 1. Client & Token Sanitation Tests
# ---------------------------------------------------------------------------
def test_client_token_sanitization():
    # Valid tokens
    client = PagerDutyClient(SAMPLE_TOKEN)
    assert repr(client) == "PagerDutyClient(api_token=<redacted>)"
    assert client._headers()["Authorization"] == f"Token token={SAMPLE_TOKEN}"

    # OAuth token
    oauth_client = PagerDutyClient("pd_oauth_secret_token_123")
    assert oauth_client._headers()["Authorization"] == "Bearer pd_oauth_secret_token_123"

    # Whitespace stripped
    clean_client = PagerDutyClient(f"  {SAMPLE_TOKEN}  ")
    assert clean_client._api_token == SAMPLE_TOKEN

    # Invalid empty or whitespace tokens
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        PagerDutyClient("")

    with pytest.raises(ValueError, match="empty or has an invalid format"):
        PagerDutyClient("token with spaces")

    with pytest.raises(ValueError, match="empty or has an invalid format"):
        PagerDutyClient("token_with_non_ascii_é")


def test_client_auth_error_handling():
    fake_http = FakeHTTPClient([FakeHTTPResponse(401, {"error": "Invalid token"})])
    client = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http, sleep=lambda _: None)

    with pytest.raises(PagerDutyAuthError, match="Check your API token"):
        client.request("GET", "incidents")


def test_client_rate_limit_and_backoff():
    fake_http = FakeHTTPClient(
        [
            FakeHTTPResponse(429, {}, headers={"Retry-After": "3"}),
            FakeHTTPResponse(200, {"incidents": []}),
        ]
    )
    sleeps: list[float] = []
    client = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http, sleep=sleeps.append)

    data = client.request("GET", "incidents")
    assert data == {"incidents": []}
    assert sleeps == [3.0]


def test_client_exhausted_retries_raises():
    fake_http = FakeHTTPClient(
        [
            FakeHTTPResponse(500, "Internal error"),
            FakeHTTPResponse(500, "Internal error"),
            FakeHTTPResponse(500, "Internal error"),
            FakeHTTPResponse(500, "Internal error"),
        ]
    )
    client = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http, sleep=lambda _: None)

    with pytest.raises(PagerDutyAPIError) as exc_info:
        client.request("GET", "incidents")
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# 2. Rendering Tests
# ---------------------------------------------------------------------------
def test_render_incident_complete():
    incident = {
        "id": "PI5678",
        "incident_number": 42,
        "title": "High memory consumption on auth-service",
        "status": "resolved",
        "urgency": "high",
        "priority": {"summary": "P1"},
        "service": {"summary": "Auth Service (PAUTH1)"},
        "escalation_policy": {"summary": "SRE Tier 1"},
        "assignments": [{"summary": "Alice Developer"}],
        "created_at": "2026-10-15T10:00:00Z",
        "last_status_change_at": "2026-10-15T11:00:00Z",
        "resolve_reason": "OOM restart resolved after scaling pods",
        "description": "Cluster node memory crossed 95% threshold.",
        "html_url": "https://example.pagerduty.com/incidents/PI5678",
    }
    notes = [
        {
            "created_at": "2026-10-15T10:15:00Z",
            "user": {"summary": "Alice Developer"},
            "content": "Scaled horizontal pod autoscaler to 8 replicas.",
        }
    ]
    log_entries = [
        {
            "created_at": "2026-10-15T10:05:00Z",
            "type": "acknowledge_log_entry",
            "summary": "Acknowledged by Alice Developer",
        }
    ]

    row = render_incident(incident, notes=notes, log_entries=log_entries)

    assert row["id"] == "incident:PI5678"
    assert row["title"] == "INC-42 High memory consumption on auth-service"
    assert row["url"] == "https://example.pagerduty.com/incidents/PI5678"
    assert row["_deleted"] is False

    content = row["content"]
    assert "Incident: INC-42" in content
    assert "Priority: P1" in content
    assert "Service: Auth Service (PAUTH1)" in content
    assert "Assigned To: Alice Developer" in content
    assert "Resolution Reason: OOM restart resolved after scaling pods" in content
    assert "Scaled horizontal pod autoscaler to 8 replicas." in content
    assert "Acknowledged by Alice Developer" in content


def test_render_incident_minimal():
    incident = {
        "id": "PIMIN",
        "summary": "Minimal incident",
        "status": "triggered",
    }
    row = render_incident(incident)

    assert row["id"] == "incident:PIMIN"
    assert row["title"] == "Minimal incident"
    assert "Status: triggered" in row["content"]
    assert "Assigned To: Unassigned" in row["content"]
    assert row["_deleted"] is False


# ---------------------------------------------------------------------------
# 3. Incremental Sync & Pagination Tests
# ---------------------------------------------------------------------------
def test_iter_rows_pagination_and_incremental():
    # Page 1: 2 incidents
    page1 = {
        "incidents": [
            {
                "id": "INC_01",
                "incident_number": 1,
                "summary": "First incident",
                "created_at": "2026-10-01T12:00:00Z",
            },
            {
                "id": "INC_02",
                "incident_number": 2,
                "summary": "Second incident",
                "created_at": "2026-10-01T12:05:00Z",
            },
        ],
        "more": False,
    }

    fake_http = FakeHTTPClient(
        [
            FakeHTTPResponse(200, page1),
            FakeHTTPResponse(200, {"notes": []}),  # for INC_01
            FakeHTTPResponse(200, {"log_entries": []}),
            FakeHTTPResponse(200, {"notes": []}),  # for INC_02
            FakeHTTPResponse(200, {"log_entries": []}),
        ]
    )
    client = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http, sleep=lambda _: None)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}

    rows = list(_iter_rows(client, state, stats, page_size=10))

    assert len(rows) == 2
    assert rows[0]["id"] == "incident:INC_01"
    assert rows[1]["id"] == "incident:INC_02"
    assert stats["scanned"] == 2
    assert stats["emitted"] == 2

    # State floor updated to the latest incident
    incidents_state = state["incidents"]
    assert incidents_state["floor"] == "2026-10-01T12:05:00Z"
    assert incidents_state["at_floor"] == ["INC_02"]

    # Second run: same data should be skipped
    fake_http_run2 = FakeHTTPClient([FakeHTTPResponse(200, page1)])
    client_run2 = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http_run2, sleep=lambda _: None)
    stats_run2: dict[str, int] = {}
    rows_run2 = list(_iter_rows(client_run2, state, stats_run2, page_size=10))
    assert isinstance(rows_run2, list)
    # INC_02 at floor was skipped, INC_01 was emitted or handled
    assert stats_run2.get("skipped", 0) >= 1


def test_iter_rows_bounded_by_max_requests():
    # Long list of pages
    page = {
        "incidents": [
            {"id": "I1", "created_at": "2026-10-01T10:00:00Z"},
        ],
        "more": True,
    }
    responses = [
        FakeHTTPResponse(200, page),
        FakeHTTPResponse(200, {"notes": []}),
        FakeHTTPResponse(200, {"log_entries": []}),
    ] * 5

    fake_http = FakeHTTPClient(responses)
    client = PagerDutyClient(SAMPLE_TOKEN, http_client=fake_http, sleep=lambda _: None)

    state: dict[str, Any] = {}
    stats: dict[str, int] = {}

    # Run with max_requests=2 should cut short cleanly without raising
    rows = list(_iter_rows(client, state, stats, max_requests=2))
    assert isinstance(rows, list)
    assert stats.get("failed_budget") == 1


# ---------------------------------------------------------------------------
# 4. Public Source Factory Contract Tests
# ---------------------------------------------------------------------------
def test_pagerduty_source_contract():
    resource = pagerduty_source(
        api_token=SAMPLE_TOKEN,
        service_ids=["PABC123"],
        include_resolved=True,
    )

    # DLT resource metadata verification
    assert getattr(resource, dlt_utils.DOCUMENT_SOURCE_ATTR) == "pagerduty"
    assert getattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR) == "pagerduty_pabc123"
    assert hasattr(resource, "cognee_sync_stats")


def test_default_resource_name():
    assert _default_resource_name(None) == "pagerduty_incidents"
    assert _default_resource_name([]) == "pagerduty_incidents"
    assert _default_resource_name(["SRV_1", "SRV_2"]) == "pagerduty_srv_1_srv_2"


def test_pagerduty_source_missing_credentials_raises():
    with pytest.raises(ValueError, match="requires an api_token"):
        pagerduty_source(api_token="")


def test_int_header_and_retry_after():
    headers = {"Retry-After": "45"}
    assert _int_header(headers, "Retry-After") == 45
    assert _retry_after(headers, 1) == 45.0

    empty_headers = {}
    assert _int_header(empty_headers, "Missing") is None
    assert _retry_after(empty_headers, 2) == 4.0
