"""Unit tests for Sentry data-source connector."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.sentry import (
    DEFAULT_SENTRY_BASE_URL,
    SentryAuthenticationError,
    SentryClient,
    SentryError,
    SentryNotFoundError,
    SentryPermissionError,
    SentryRateLimitError,
    _extract_stacktrace,
    _mask_token,
    _parse_link_header,
    _render_sentry_issue_markdown,
    build_sentry_service,
    sentry_source,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)


def _make_fake_response(data, headers=None, status=200):
    """Helper to mock urllib response context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_resp.headers = headers or {}
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


def test_client_token_validation():
    """Verify authentication token validation and masking."""
    with pytest.raises(SentryAuthenticationError):
        SentryClient(auth_token="")

    with pytest.raises(SentryAuthenticationError):
        SentryClient(auth_token="   ")

    client = SentryClient(auth_token="sntrys_1234567890abcdef")
    assert _mask_token("sntrys_1234567890abcdef") == "sntr...cdef"
    assert _mask_token("short") == "sentry_***"
    assert "sntrys_1234567890abcdef" not in repr(client)
    assert client.base_url == DEFAULT_SENTRY_BASE_URL


def test_client_headers():
    """Verify Sentry headers include Authorization and User-Agent."""
    client = SentryClient(auth_token="test-token-xyz")
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer test-token-xyz"
    assert headers["Accept"] == "application/json"
    assert "cognee-sentry-connector" in headers["User-Agent"]


def test_client_parse_errors():
    """Verify HTTP errors map to specific Sentry exceptions."""
    client = SentryClient(auth_token="test-token")

    # 401
    err_401 = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    with (
        patch("urllib.request.urlopen", side_effect=err_401),
        pytest.raises(SentryAuthenticationError),
    ):
        client.request("api/0/projects/org/proj/issues/")

    # 403
    err_403 = urllib.error.HTTPError("url", 403, "Forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=err_403), pytest.raises(SentryPermissionError):
        client.request("api/0/projects/org/proj/issues/")

    # 404
    err_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=err_404), pytest.raises(SentryNotFoundError):
        client.request("api/0/projects/org/proj/issues/")

    # 429
    err_429 = urllib.error.HTTPError("url", 429, "Too Many Requests", {"Retry-After": "30"}, None)
    with (
        patch("urllib.request.urlopen", side_effect=err_429),
        pytest.raises(SentryRateLimitError) as exc_info,
    ):
        client.request("api/0/projects/org/proj/issues/")
    assert "Retry-After: 30" in str(exc_info.value)


def test_link_header_parser():
    """Verify RFC-5988 Link header extraction of next cursor and has_more flag."""
    link_header = (
        '<https://sentry.io/api/0/projects/org/proj/issues/?cursor=100:0:1>; rel="previous"; results="false", '
        '<https://sentry.io/api/0/projects/org/proj/issues/?cursor=100:0:0>; rel="next"; results="true"'
    )
    cursor, has_more = _parse_link_header(link_header)
    assert cursor == "100:0:0"
    assert has_more is True

    # When results="false" for next
    link_header_end = (
        '<https://sentry.io/api/0/projects/org/proj/issues/?cursor=100:0:1>; rel="previous"; results="true", '
        '<https://sentry.io/api/0/projects/org/proj/issues/?cursor=100:0:0>; rel="next"; results="false"'
    )
    cursor_end, has_more_end = _parse_link_header(link_header_end)
    assert cursor_end == "100:0:0"
    assert has_more_end is False

    # None header
    c, m = _parse_link_header(None)
    assert c is None
    assert m is False


def test_render_sentry_issue_markdown():
    """Verify deterministic markdown formatting of Sentry incident document."""
    issue = {
        "id": "1048291",
        "title": "TypeError: Cannot read properties of undefined",
        "culprit": "src/auth/jwt.py in verify_token",
        "status": "unresolved",
        "level": "error",
        "count": "4210",
        "userCount": "312",
        "firstSeen": "2026-10-18T10:00:00Z",
        "lastSeen": "2026-10-19T23:30:00Z",
        "permalink": "https://sentry.io/organizations/acme/issues/1048291/",
    }
    latest_event = {
        "tags": [
            {"key": "environment", "value": "production"},
            {"key": "release", "value": "v1.4.2"},
        ],
        "entries": [
            {
                "type": "exception",
                "data": {
                    "values": [
                        {
                            "type": "TypeError",
                            "value": "Cannot read properties of undefined",
                            "stacktrace": {
                                "frames": [
                                    {
                                        "filename": "src/auth/jwt.py",
                                        "lineNo": 42,
                                        "function": "verify_token",
                                        "context": [[42, "user = payload['user']"]],
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ],
    }

    md = _render_sentry_issue_markdown(issue, latest_event, "acme", "backend-service")

    assert "# Sentry Incident: TypeError: Cannot read properties of undefined" in md
    assert "- **Issue ID**: `1048291`" in md
    assert "- **Project**: `acme/backend-service`" in md
    assert "- **Occurrences**: `4210`" in md
    assert "- `environment=production`" in md
    assert "- `release=v1.4.2`" in md
    assert 'File "src/auth/jwt.py", line 42, in verify_token' in md
    assert "user = payload['user']" in md


def test_extract_stacktrace_formatting():
    """Verify extraction and formatting of exception frames from event entries."""
    event = {
        "entries": [
            {
                "type": "exception",
                "data": {
                    "values": [
                        {
                            "type": "KeyError",
                            "value": "'missing_key'",
                            "stacktrace": {
                                "frames": [
                                    {
                                        "filename": "app/main.py",
                                        "lineNo": 12,
                                        "function": "process_event",
                                        "context": [],
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ]
    }
    frames = _extract_stacktrace(event)
    assert any('File "app/main.py", line 12, in process_event' in f for f in frames)
    assert any("KeyError: 'missing_key'" in f for f in frames)

    assert _extract_stacktrace(None) == []


def test_sentry_source_full_crawl():
    """Verify sentry_source yields normalized documents with node set scoping."""
    client = SentryClient(auth_token="dummy-token")

    fake_issues = [
        {
            "id": "1001",
            "title": "NullPointerException in user_service",
            "culprit": "services/user.py",
            "status": "unresolved",
            "level": "error",
            "count": "5",
            "lastSeen": "2026-10-19T20:00:00Z",
        }
    ]

    mock_resp_issues = _make_fake_response(fake_issues, headers={"link": ""})
    mock_resp_event = _make_fake_response(
        {"tags": [{"key": "env", "value": "staging"}], "entries": []}
    )

    with patch("urllib.request.urlopen", side_effect=[mock_resp_issues, mock_resp_event]):
        src = sentry_source(
            organization_slug="my-org",
            project_slug="my-proj",
            client=client,
            fetch_events=True,
        )

        assert getattr(src, DOCUMENT_SOURCE_ATTR) == "sentry"
        assert getattr(src, PIPELINE_SCOPE_ATTR) == "sentry:my-org/my-proj"

        resource = src.selected_resources["sentry_issues"]
        rows = list(resource)

        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == "sentry:my-proj:issue:1001"
        assert row["status"] == "unresolved"
        assert row[NODE_SET_COLUMN] == ["sentry:my-org/my-proj"]
        assert "NullPointerException" in row["text"]


def test_sentry_source_rate_limit_cut_short():
    """Verify crawler yields current batch and gracefully exits when rate limit is near floor."""
    client = SentryClient(auth_token="dummy-token")

    fake_issues = [
        {
            "id": "2001",
            "title": "RedisConnectionError",
            "culprit": "cache/redis.py",
            "status": "unresolved",
            "count": "20",
        }
    ]

    # Return remaining = "2" (below floor of 5)
    mock_resp = _make_fake_response(
        fake_issues,
        headers={
            "x-sentry-rate-limit-remaining": "2",
            "link": '<https://sentry.io/.../?cursor=200:0:0>; rel="next"; results="true"',
        },
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        src = sentry_source(
            organization_slug="acme",
            project_slug="backend",
            client=client,
            fetch_events=False,
        )
        resource = src.selected_resources["sentry_issues"]
        rows = list(resource)

        # Batch is yielded despite rate limit
        assert len(rows) == 1
        assert rows[0]["id"] == "sentry:backend:issue:2001"


def test_sentry_source_factory():
    """Verify build_sentry_service and factory defaults."""
    client = build_sentry_service("token-abc", base_url="https://sentry.example.com")
    assert client.base_url == "https://sentry.example.com"

    src = sentry_source(
        organization_slug="test-org",
        project_slug="test-proj",
        auth_token="token-abc",
    )
    assert getattr(src, DOCUMENT_SOURCE_ATTR) == "sentry"


def test_sentry_source_incremental_state():
    """Verify crawler receives initial cursor from dlt resource state."""
    client = SentryClient(auth_token="dummy-token")
    fake_issues = [{"id": "5555", "title": "IndexError", "culprit": "app.py", "status": "resolved"}]
    mock_resp = _make_fake_response(fake_issues, headers={"link": ""})

    with (
        patch("dlt.current.resource_state", return_value={"cursor": "cursor_12345"}),
        patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
    ):
        src = sentry_source(
            organization_slug="org-x",
            project_slug="proj-y",
            client=client,
            fetch_events=False,
        )
        resource = src.selected_resources["sentry_issues"]
        rows = list(resource)

        assert len(rows) == 1
        assert rows[0]["id"] == "sentry:proj-y:issue:5555"
        # Verify the request included the cursor from state
        called_req = mock_urlopen.call_args[0][0]
        assert "cursor=cursor_12345" in called_req.full_url
