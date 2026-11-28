from unittest.mock import MagicMock

import pytest
import requests

from cognee.tasks.ingestion.connectors.linkding import (
    LinkdingAuthError,
    LinkdingClient,
    _iter_rows,
    linkding_source,
)


@pytest.fixture
def mock_bookmarks():
    return [
        {
            "id": 1,
            "url": "https://docs.python.org/3/",
            "title": "Python 3 Documentation",
            "description": "Official Python docs and language reference",
            "notes": "Essential reference for core features",
            "website_title": "Python Documentation",
            "is_archived": False,
            "unread": False,
            "shared": False,
            "tag_names": ["python", "docs"],
            "date_added": "2024-01-01T10:00:00Z",
            "date_modified": "2024-01-02T12:00:00Z",
        },
        {
            "id": 2,
            "url": "https://fastapi.tiangolo.com/",
            "title": "FastAPI",
            "description": "Modern, fast web framework for Python",
            "notes": "Good for building async APIs",
            "website_title": "FastAPI framework",
            "is_archived": False,
            "unread": True,
            "shared": False,
            "tag_names": ["python", "api"],
            "date_added": "2024-01-03T10:00:00Z",
            "date_modified": "2024-01-04T15:00:00Z",
        },
    ]


class TestLinkdingConnector:
    def test_linkding_basic_sync(self, mock_bookmarks):
        client = LinkdingClient("https://bookmarks.example.com", "test-token")

        def mock_request(method, url, params=None, **kwargs):
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            if "/api/bookmarks/" in url:
                resp.json.return_value = {
                    "count": len(mock_bookmarks),
                    "next": None,
                    "previous": None,
                    "results": mock_bookmarks,
                }
            return resp

        client.request = MagicMock(side_effect=mock_request)

        state = {}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        rows = list(_iter_rows(client, state, stats))

        assert len(rows) == 2
        assert stats["synced"] == 2
        assert stats["deleted"] == 0
        assert state["cursor"] == "2024-01-04T15:00:00Z"
        assert state["members"] == [1, 2]

        r1 = rows[0]
        assert r1["id"] == "linkding:bookmark:1"
        assert r1["title"] == "Python 3 Documentation"
        assert "Essential reference for core features" in r1["content"]
        assert "linkding:tag:python" in r1["cognee_node_set"]
        assert "linkding:tag:docs" in r1["cognee_node_set"]

    def test_linkding_incremental_cursor(self, mock_bookmarks):
        client = LinkdingClient("https://bookmarks.example.com", "test-token")
        requested_urls = []

        def mock_request(method, url, params=None, **kwargs):
            requested_urls.append((url, params))
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            resp.json.return_value = {
                "count": 0,
                "next": None,
                "previous": None,
                "results": [],
            }
            return resp

        client.request = MagicMock(side_effect=mock_request)

        state = {"cursor": "2024-01-04T15:00:00Z", "members": [1, 2]}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        list(_iter_rows(client, state, stats))

        # Check that modified_since was passed in params
        first_call = requested_urls[0]
        assert first_call[1]["modified_since"] == "2024-01-04T15:00:00Z"

    def test_linkding_forget_on_delete_with_404(self, mock_bookmarks):
        client = LinkdingClient("https://bookmarks.example.com", "test-token")

        # Bookmark 2 was deleted from Linkding
        active_bookmarks = [mock_bookmarks[0]]

        def mock_request(method, url, params=None, **kwargs):
            resp = MagicMock(spec=requests.Response)
            if url.endswith("/api/bookmarks/2/"):
                resp.status_code = 404
                return resp

            resp.status_code = 200
            resp.json.return_value = {
                "count": len(active_bookmarks),
                "next": None,
                "previous": None,
                "results": active_bookmarks,
            }
            return resp

        client.request = MagicMock(side_effect=mock_request)

        state = {"cursor": "2024-01-04T15:00:00Z", "members": [1, 2]}
        stats = {"synced": 0, "deleted": 0, "failed": 0}
        rows = list(_iter_rows(client, state, stats))

        # 1 synced row + 1 tombstone
        tombstones = [r for r in rows if r.get("_deleted") is True]
        assert len(tombstones) == 1
        assert tombstones[0]["id"] == "linkding:bookmark:2"
        assert stats["deleted"] == 1
        assert state["members"] == [1]

    def test_linkding_auth_error(self):
        client = LinkdingClient("https://bookmarks.example.com", "bad-token")

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 401
        client.session.request = MagicMock(return_value=mock_resp)

        with pytest.raises(LinkdingAuthError, match="Invalid Linkding API token"):
            client.request("GET", "/api/bookmarks/")

    def test_linkding_source_factory_validation(self):
        with pytest.raises(ValueError, match="base_url must be a non-empty string"):
            linkding_source(base_url="", api_token="secret")

        with pytest.raises(ValueError, match="api_token must be a non-empty string"):
            linkding_source(base_url="https://example.com", api_token="")

        source = linkding_source(
            base_url="https://bookmarks.example.com/",
            api_token="valid_token",
            resource_name="my_bookmarks",
        )
        assert source.name == "my_bookmarks"
        assert source.cognee_document_source == "linkding"
