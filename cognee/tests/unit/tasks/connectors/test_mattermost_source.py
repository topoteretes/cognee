"""Tests for Mattermost data-source connector: authentication, thread clustering, incremental sync, and tombstones."""

import os
from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.ingestion.connectors.mattermost import (
    MattermostAPIError,
    MattermostAuthError,
    MattermostClient,
    MattermostRateLimitedError,
    MattermostResourceNotFoundError,
    MattermostSettings,
    _is_edit_audit_clone,
    _is_system_post,
    _iter_channel_rows,
    _render_thread_markdown,
    mattermost_source,
)


class MockHTTPResponse:
    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        self._data = data
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_client_validation():
    with pytest.raises(ValueError, match="base_url must not be empty"):
        MattermostClient(base_url="", token="pat_secret")

    with pytest.raises(ValueError, match="token must not be empty"):
        MattermostClient(base_url="https://chat.example.com", token=" ")


def test_client_token_masking():
    client = MattermostClient(base_url="https://chat.company.com", token="super_secret_pat_token")
    rep = repr(client)
    assert "super_secret_pat_token" not in rep
    assert "***" in rep
    assert "https://chat.company.com" in rep


def test_client_headers():
    client = MattermostClient(base_url="https://chat.company.com", token="my_token_123")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer my_token_123"
    assert headers["Accept"] == "application/json"
    assert "cognee-mattermost-connector" in headers["User-Agent"]


def test_client_auth_error_handling():
    import urllib.error

    client = MattermostClient(base_url="https://chat.example.com", token="bad_token")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://chat.example.com/api/v4/users/me",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(MattermostAuthError, match="authentication failed \\(HTTP 401\\)"),
    ):
        client.get_me()


def test_client_resource_not_found():
    import urllib.error

    client = MattermostClient(base_url="https://chat.example.com", token="token")

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://chat.example.com/api/v4/channels/nonexistent",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    with (
        patch("urllib.request.urlopen", side_effect=mock_urlopen),
        pytest.raises(MattermostResourceNotFoundError, match="resource not found \\(HTTP 404\\)"),
    ):
        client._request("channels/nonexistent")


def test_client_rate_limited():
    import urllib.error

    client = MattermostClient(base_url="https://chat.example.com", token="token", max_retries=1)

    def mock_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://chat.example.com/api/v4/posts",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "5"},
            fp=None,
        )

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        with pytest.raises(MattermostRateLimitedError) as exc_info:
            client._request("posts")
        assert exc_info.value.retry_after == 5


def test_user_caching():
    client = MattermostClient(base_url="https://chat.example.com", token="token")

    call_count = 0

    def mock_request(endpoint, params=None):
        nonlocal call_count
        call_count += 1
        return {"id": "user123", "username": "alice"}

    client._request = mock_request

    name1 = client.get_username("user123")
    name2 = client.get_username("user123")
    assert name1 == "alice"
    assert name2 == "alice"
    assert call_count == 1  # Second call served from cache


def test_system_post_and_audit_clone_filtering():
    system_post = {"id": "p1", "type": "system_join_channel", "message": "@alice joined"}
    normal_post = {"id": "p2", "type": "", "message": "Hello team"}
    audit_clone = {"id": "p3", "type": "", "original_id": "p2", "message": "Old message"}

    assert _is_system_post(system_post) is True
    assert _is_system_post(normal_post) is False

    assert _is_edit_audit_clone(audit_clone) is True
    assert _is_edit_audit_clone(normal_post) is False


def test_render_thread_markdown():
    root = {
        "id": "root_999",
        "user_id": "u1",
        "create_at": 1700000000000,
        "update_at": 1700000010000,
        "message": "Incident: Payment gateway timeout error",
    }
    replies = [
        {
            "id": "rep_2",
            "root_id": "root_999",
            "user_id": "u2",
            "create_at": 1700000005000,
            "message": "Investigating logs now.",
        },
        {
            "id": "rep_1",
            "root_id": "root_999",
            "user_id": "u1",
            "create_at": 1700000002000,
            "message": "Triggered by deploy hash #a8f3.",
        },
    ]
    channel_info = {"id": "ch1", "name": "incidents", "display_name": "Production Incidents"}

    def mock_resolve_user(uid):
        return "alice" if uid == "u1" else "bob"

    md = _render_thread_markdown(
        root_post=root,
        replies=replies,
        channel_info=channel_info,
        server_host="chat.company.com",
        resolve_username=mock_resolve_user,
    )

    assert "# Mattermost: #incidents - Thread root_999" in md
    assert "- **Channel**: `#incidents` (Production Incidents)" in md
    assert "- **Author**: @alice" in md
    assert "- **Total Replies**: 2" in md
    assert "Incident: Payment gateway timeout error" in md
    # Chronological sort check: rep_1 (timestamp ..002000) should appear before rep_2 (timestamp ..005000)
    pos_rep1 = md.find("Triggered by deploy hash #a8f3.")
    pos_rep2 = md.find("Investigating logs now.")
    assert pos_rep1 != -1 and pos_rep2 != -1
    assert pos_rep1 < pos_rep2


def test_iter_channel_rows_thread_reconstruction_and_cursor():
    client = MagicMock(spec=MattermostClient)
    client.base_url = "https://chat.company.com"
    client.get_username.side_effect = lambda uid: "alice" if uid == "u1" else "bob"

    # Mock posts returned by channel
    client.get_channel_posts.return_value = {
        "order": ["p_root", "p_reply"],
        "posts": {
            "p_root": {
                "id": "p_root",
                "root_id": "",
                "user_id": "u1",
                "create_at": 1000,
                "update_at": 2000,
                "delete_at": 0,
                "message": "Let's migrate to Kuzu graph engine.",
            },
            "p_reply": {
                "id": "p_reply",
                "root_id": "p_root",
                "user_id": "u2",
                "create_at": 1500,
                "update_at": 2000,
                "delete_at": 0,
                "message": "Benchmark tests look great.",
            },
        },
    }

    # Mock get_post_thread
    client.get_post_thread.return_value = {
        "order": ["p_root", "p_reply"],
        "posts": {
            "p_root": {
                "id": "p_root",
                "root_id": "",
                "user_id": "u1",
                "create_at": 1000,
                "update_at": 2000,
                "delete_at": 0,
                "message": "Let's migrate to Kuzu graph engine.",
            },
            "p_reply": {
                "id": "p_reply",
                "root_id": "p_root",
                "user_id": "u2",
                "create_at": 1500,
                "update_at": 2000,
                "delete_at": 0,
                "message": "Benchmark tests look great.",
            },
        },
    }

    channel = {"id": "ch_eng", "name": "engineering"}
    channel_state = {"cursor": 0, "seen_ids": []}
    stats = {}

    rows = list(
        _iter_channel_rows(
            client=client,
            channel=channel,
            channel_state=channel_state,
            stats=stats,
            server_host="chat.company.com",
        )
    )

    assert len(rows) == 1
    doc = rows[0]
    assert doc["id"] == "mattermost:thread:p_root"
    assert doc["_deleted"] is False
    assert "Let's migrate to Kuzu graph engine." in doc["content"]
    assert "Benchmark tests look great." in doc["content"]
    assert doc["cognee_node_set"] == ["mattermost:chat.company.com/engineering"]

    # Cursor must advance to 2000
    assert channel_state["cursor"] == 2000
    assert "p_root" in channel_state["seen_ids"]
    assert "p_reply" in channel_state["seen_ids"]


def test_iter_channel_rows_deleted_tombstone():
    client = MagicMock(spec=MattermostClient)
    client.base_url = "https://chat.company.com"

    # Channel returns a deleted root post
    client.get_channel_posts.return_value = {
        "order": ["p_deleted"],
        "posts": {
            "p_deleted": {
                "id": "p_deleted",
                "root_id": "",
                "create_at": 1000,
                "update_at": 3000,
                "delete_at": 3000,  # Deleted
                "message": "",
            }
        },
    }

    channel = {"id": "ch_eng", "name": "engineering"}
    channel_state = {"cursor": 1000, "seen_ids": []}
    stats = {}

    rows = list(
        _iter_channel_rows(
            client=client,
            channel=channel,
            channel_state=channel_state,
            stats=stats,
            server_host="chat.company.com",
        )
    )

    assert len(rows) == 1
    tombstone = rows[0]
    assert tombstone["id"] == "mattermost:thread:p_deleted"
    assert tombstone["_deleted"] is True
    assert stats["deleted"] == 1


def test_mattermost_settings_env_loading(monkeypatch):
    monkeypatch.setenv("MATTERMOST_URL", "https://custom.mattermost.com")
    monkeypatch.setenv("MATTERMOST_BOT_TOKEN", "pat_test_token_99")
    monkeypatch.setenv("MATTERMOST_CHANNELS", '["general", "dev"]')

    settings = MattermostSettings()
    assert settings.mattermost_url == "https://custom.mattermost.com"
    assert settings.mattermost_token == "pat_test_token_99"
    assert settings.mattermost_channels == ["general", "dev"]


def test_mattermost_source_full_crawl():
    client = MagicMock(spec=MattermostClient)
    client.base_url = "https://chat.company.com"
    client.list_my_channels.return_value = [
        {"id": "c1", "name": "general", "display_name": "General Chat"}
    ]
    client.get_channel_posts.return_value = {
        "order": ["post_alpha"],
        "posts": {
            "post_alpha": {
                "id": "post_alpha",
                "root_id": "",
                "user_id": "u1",
                "create_at": 5000,
                "update_at": 5000,
                "delete_at": 0,
                "message": "Welcome to the new release!",
            }
        },
    }
    client.get_post_thread.return_value = {
        "order": ["post_alpha"],
        "posts": {
            "post_alpha": {
                "id": "post_alpha",
                "root_id": "",
                "user_id": "u1",
                "create_at": 5000,
                "update_at": 5000,
                "delete_at": 0,
                "message": "Welcome to the new release!",
            }
        },
    }
    client.get_username.return_value = "charlie"

    source = mattermost_source(
        client=client,
        channels=["general"],
    )

    items = list(source)
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "mattermost:thread:post_alpha"
    assert item["title"] == "Mattermost: #general - Thread post_alpha"
    assert item["_deleted"] is False
    assert "Welcome to the new release!" in item["content"]
