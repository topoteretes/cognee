"""JSON Feed connector unit tests: HTML sanitization, feed parsing, pagination, incremental cursor, and dlt source."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from cognee.tasks.ingestion.connectors.jsonfeed import (
    JSONFeedFetchError,
    JSONFeedParseError,
    JSONFeedService,
    build_jsonfeed_service,
    html_to_text,
    jsonfeed_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN, PIPELINE_SCOPE_ATTR

MOCK_FEED_V1_1 = {
    "version": "https://jsonfeed.org/version/1.1",
    "title": "Engineering Blog",
    "home_page_url": "https://example.com",
    "feed_url": "https://example.com/feed.json",
    "items": [
        {
            "id": "item-101",
            "url": "https://example.com/posts/item-101",
            "title": "Introducing JSON Feed Support",
            "content_html": "<p>We are thrilled to announce <b>JSON Feed</b> integration in Cognee!</p>",
            "summary": "Announcing JSON Feed support in Cognee.",
            "date_published": "2026-03-01T10:00:00Z",
            "date_modified": "2026-03-02T12:00:00Z",
            "authors": [{"name": "Jane Doe"}],
            "tags": ["releases", "integrations"],
        },
        {
            "id": "item-100",
            "url": "https://example.com/posts/item-100",
            "title": "Initial Announcement",
            "content_text": "Plain text content without HTML tags.",
            "date_published": "2026-02-01T08:00:00Z",
            "tags": ["general"],
        },
    ],
}


def test_html_to_text_sanitizes_html():
    raw_html = "<p>Hello <b>World</b> &amp; universe!</p><div>Test <span>line</span>.</div>"
    result = html_to_text(raw_html)
    assert "Hello World & universe!" in result
    assert "Test line." in result
    assert "<" not in result
    assert ">" not in result


def test_html_to_text_handles_none_and_empty():
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


def test_jsonfeed_service_fetch_feed_success():
    service = JSONFeedService("https://example.com/feed.json")

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(MOCK_FEED_V1_1).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        feed = service.fetch_feed("https://example.com/feed.json")

    assert feed["title"] == "Engineering Blog"
    assert len(feed["items"]) == 2


def test_jsonfeed_service_fetch_feed_invalid_json():
    service = JSONFeedService("https://example.com/feed.json")

    mock_resp = MagicMock()
    mock_resp.read.return_value = b"invalid json content <<<"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(JSONFeedParseError):
            service.fetch_feed("https://example.com/feed.json")


def test_jsonfeed_format_item_document():
    service = JSONFeedService("https://example.com/feed.json")
    item = MOCK_FEED_V1_1["items"][0]

    doc = service.format_item_document(item, MOCK_FEED_V1_1, "https://example.com/feed.json")

    assert doc["id"] == "jsonfeed:https://example.com/feed.json#item-101"
    assert doc["title"] == "Introducing JSON Feed Support"
    assert doc["url"] == "https://example.com/posts/item-101"
    assert doc["_deleted"] is False
    assert "jsonfeed" in doc[NODE_SET_COLUMN]
    assert "jsonfeed:engineering_blog" in doc[NODE_SET_COLUMN]

    content = doc["content"]
    assert "# Introducing JSON Feed Support" in content
    assert "Engineering Blog" in content
    assert "Jane Doe" in content
    assert "releases, integrations" in content
    assert "We are thrilled to announce JSON Feed integration in Cognee!" in content


def test_jsonfeed_service_incremental_filtering_and_pagination():
    page1 = {
        "version": "https://jsonfeed.org/version/1.1",
        "title": "Blog",
        "next_url": "https://example.com/feed-page2.json",
        "items": [
            {
                "id": "p1-newer",
                "title": "Newer Post",
                "content_text": "Newer post body",
                "date_published": "2026-03-05T00:00:00Z",
            },
            {
                "id": "p1-older",
                "title": "Older Post",
                "content_text": "Older post body",
                "date_published": "2026-02-01T00:00:00Z",
            },
        ],
    }
    page2 = {
        "version": "https://jsonfeed.org/version/1.1",
        "title": "Blog",
        "items": [
            {
                "id": "p2-oldest",
                "title": "Oldest Post",
                "content_text": "Oldest post body",
                "date_published": "2026-01-01T00:00:00Z",
            }
        ],
    }

    service = JSONFeedService("https://example.com/feed.json")

    def mock_fetch(url):
        if "page2" in url:
            return page2
        return page1

    with patch.object(service, "fetch_feed", side_effect=mock_fetch):
        # Full sync: should get all 3 posts
        docs_all, newest = service.get_feed_items("https://example.com/feed.json")
        assert len(docs_all) == 3
        assert newest == "2026-03-05T00:00:00Z"

        # Incremental sync: cursor is 2026-02-15 -> only "p1-newer" should be returned
        docs_inc, newest_inc = service.get_feed_items(
            "https://example.com/feed.json",
            since_iso="2026-02-15T00:00:00Z",
        )
        assert len(docs_inc) == 1
        assert docs_inc[0]["title"] == "Newer Post"
        assert newest_inc == "2026-03-05T00:00:00Z"


def test_build_jsonfeed_service_normalizes():
    s = build_jsonfeed_service(["https://example.com/feed1.json ", "https://example.com/feed2.json"])
    assert len(s.feed_urls) == 2
    assert s.feed_urls[0] == "https://example.com/feed1.json"


def test_jsonfeed_source_metadata():
    pytest.importorskip("dlt")
    source = jsonfeed_source(feed_urls=["https://example.com/feed.json"])

    assert getattr(source, DOCUMENT_SOURCE_ATTR) is True
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "jsonfeed"
    assert "items" in source.resources
