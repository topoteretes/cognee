"""Tests for Fathom meeting notes data-source connector: authentication, summaries, action items, and deletion reconciliation."""

from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.fathom import (
    FathomAuthError,
    FathomClient,
    FathomRateLimitedError,
    FathomResourceNotFoundError,
    FathomSettings,
    _format_seconds,
    _iter_fathom_rows,
    _render_meeting_markdown,
    fathom_source,
)


def test_client_validation():
    with pytest.raises(ValueError, match="api_key must not be empty"):
        FathomClient(api_key="")

    with pytest.raises(ValueError, match="api_key must not be empty"):
        FathomClient(api_key="   ")


def test_client_key_masking():
    client = FathomClient(api_key="secret_fathom_key_999")
    rep = repr(client)
    assert "secret_fathom_key_999" not in rep
    assert "***" in rep
    assert "https://api.fathom.video/external/v1" in rep


def test_client_headers():
    client = FathomClient(api_key="my_fathom_token")
    headers = client._headers()
    assert headers["X-Api-Key"] == "my_fathom_token"
    assert headers["Accept"] == "application/json"
    assert "cognee-fathom-connector" in headers["User-Agent"]


def test_client_auth_error_handling():
    import urllib.error

    client = FathomClient(api_key="invalid_token")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.fathom.video/external/v1/meetings",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(FathomAuthError, match="authentication failed \\(HTTP 401\\)"),
    ):
        client.list_meetings()


def test_client_resource_not_found():
    import urllib.error

    client = FathomClient(api_key="key")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.fathom.video/external/v1/meetings/missing",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(FathomResourceNotFoundError, match="resource not found \\(HTTP 404\\)"),
    ):
        client._request("meetings/missing")


def test_client_rate_limited():
    import urllib.error

    client = FathomClient(api_key="key", max_retries=1)

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.fathom.video/external/v1/meetings",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "6"},
            fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(FathomRateLimitedError) as exc_info:
            client.list_meetings()
        assert exc_info.value.retry_after == 6


def test_format_seconds():
    assert _format_seconds(45) == "00:45"
    assert _format_seconds(125) == "02:05"
    assert _format_seconds(3665) == "01:01:05"
    assert _format_seconds(None) == "00:00"


def test_render_meeting_markdown():
    meeting = {
        "id": 101,
        "title": "Engineering Sprint Planning",
        "url": "https://fathom.video/share/meet_101",
        "created_at": "2026-10-18T10:00:00Z",
        "recording_start_time": "2026-10-18T10:01:00Z",
        "duration": 1800,
        "invitees": [
            {"name": "Alice Developer", "email": "alice@company.com"},
            {"name": "Bob Team Lead", "email": "bob@company.com"},
        ],
        "default_summary": {
            "overview": "Reviewed the Q4 roadmap and finalized data connector integrations.",
            "key_points": [
                "Agreed to ship Sentry, Mattermost, and PandaDoc connectors",
                "Assigned test benchmarks to Alice",
            ],
        },
        "action_items": [
            {
                "description": "Benchmark graph query response times",
                "assignee": {"name": "Alice"},
                "completed": False,
            },
            {
                "description": "Open PR for Fathom connector",
                "assignee": {"name": "Bob"},
                "completed": True,
            },
        ],
        "transcript": [
            {
                "speaker": {"name": "Bob"},
                "start_time": 10.5,
                "text": "Good morning team, let's look at the open tasks.",
            },
            {
                "speaker": {"name": "Alice"},
                "start_time": 25.0,
                "text": "The tests are running smoothly and passing in under a second.",
            },
        ],
    }

    # Test without transcript
    md_summary = _render_meeting_markdown(meeting, include_transcript=False)
    assert "# Meeting: Engineering Sprint Planning" in md_summary
    assert "- **Meeting ID**: `101`" in md_summary
    assert "- **Duration**: 30:00" in md_summary
    assert "- **Attendees**: Alice Developer, Bob Team Lead" in md_summary
    assert "Reviewed the Q4 roadmap and finalized data connector integrations." in md_summary
    assert (
        "- [ ] **Benchmark graph query response times** (Assignee: @Alice, Status: Pending)"
        in md_summary
    )
    assert (
        "- [x] **Open PR for Fathom connector** (Assignee: @Bob, Status: Completed)" in md_summary
    )
    assert "Dialogue & Discussion Transcript" not in md_summary

    # Test with transcript enabled
    md_full = _render_meeting_markdown(meeting, include_transcript=True)
    assert "Dialogue & Discussion Transcript" in md_full
    assert "- **Bob** *(00:10)*: Good morning team, let's look at the open tasks." in md_full
    assert (
        "- **Alice** *(00:25)*: The tests are running smoothly and passing in under a second."
        in md_full
    )


def test_iter_fathom_rows_ingest_and_cursor():
    client = MagicMock(spec=FathomClient)
    client.list_meetings.return_value = {
        "items": [
            {
                "id": 555,
                "title": "Design Review",
                "created_at": "2026-10-18T15:00:00Z",
                "recording_start_time": "2026-10-18T15:01:00Z",
                "duration": 900,
                "url": "https://fathom.video/share/555",
                "default_summary": {"overview": "Approved UI layout."},
                "action_items": [],
            }
        ],
        "next_cursor": None,
    }

    state = {"cursor": None, "seen_ids": []}
    stats = {}

    rows = list(
        _iter_fathom_rows(
            client=client,
            include_transcript=False,
            recheck_days=7,
            state=state,
            stats=stats,
        )
    )

    assert len(rows) == 1
    doc = rows[0]
    assert doc["id"] == "fathom:meeting:555"
    assert doc["_deleted"] is False
    assert "Meeting: Design Review" in doc["title"]
    assert "Approved UI layout." in doc["content"]
    assert doc["cognee_node_set"] == ["fathom:meetings"]

    assert state["cursor"] == "2026-10-18T15:00:00Z"
    assert "555" in state["seen_ids"]


def test_iter_fathom_rows_deletion_reconciliation():
    client = MagicMock(spec=FathomClient)
    # Only meeting 200 is returned; meeting 100 was deleted in Fathom
    client.list_meetings.return_value = {
        "items": [
            {
                "id": 200,
                "title": "Weekly Standup",
                "created_at": "2026-10-18T16:00:00Z",
            }
        ],
        "next_cursor": None,
    }

    # State has previously seen meeting 100
    state = {"cursor": None, "seen_ids": ["100"]}
    stats = {}

    rows = list(
        _iter_fathom_rows(
            client=client,
            include_transcript=False,
            recheck_days=7,
            state=state,
            stats=stats,
        )
    )

    # Should yield active meeting 200, then tombstone for deleted meeting 100
    assert len(rows) == 2
    active_row = rows[0]
    assert active_row["id"] == "fathom:meeting:200"
    assert active_row["_deleted"] is False

    tombstone = rows[1]
    assert tombstone["id"] == "fathom:meeting:100"
    assert tombstone["_deleted"] is True
    assert stats["deleted"] == 1


def test_fathom_settings_env_loading(monkeypatch):
    monkeypatch.setenv("FATHOM_API_KEY", "env_fathom_key_123")
    monkeypatch.setenv("FATHOM_BASE_URL", "https://api.fathom.video/external/v1")
    monkeypatch.setenv("FATHOM_INCLUDE_TRANSCRIPT", "true")

    settings = FathomSettings()
    assert settings.fathom_api_key == "env_fathom_key_123"
    assert settings.fathom_base_url == "https://api.fathom.video/external/v1"
    assert settings.fathom_include_transcript is True


def test_fathom_source_full_crawl():
    client = MagicMock(spec=FathomClient)
    client.list_meetings.return_value = {
        "items": [
            {
                "id": 777,
                "title": "Architecture Deep Dive",
                "created_at": "2026-10-18T18:00:00Z",
                "default_summary": {"overview": "Kuzu graph engine selected."},
                "action_items": [],
            }
        ],
        "next_cursor": None,
    }

    source = fathom_source(
        client=client,
        include_transcript=False,
    )

    items = list(source)
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "fathom:meeting:777"
    assert item["title"] == "Meeting: Architecture Deep Dive"
    assert item["_deleted"] is False
    assert "Kuzu graph engine selected." in item["content"]
