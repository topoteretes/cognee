"""Tests for the Bluesky (AT Protocol) DLT connector:
facet byte offset resolution, thread flattening, error mapping, token hygiene,
and incremental sync invariants.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from cognee.tasks.ingestion.connectors import bluesky as bluesky_module
from cognee.tasks.ingestion.connectors.bluesky import (
    BlueskyActorNotFoundError,
    BlueskyAPIError,
    BlueskyAuthError,
    BlueskyClient,
    BlueskyRateLimitedError,
    _iter_rows,
    bluesky_source,
    extract_post_web_url,
    render_post_document,
    resolve_facets,
)


class FakeResponse:
    def __init__(
        self,
        json_data: Any,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        self._json_data = json_data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeSession:
    def __init__(self, responses: list[FakeResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: Any = None,
    ) -> FakeResponse:
        self.calls.append((url, params or {}, headers or {}))
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse({"feed": [], "cursor": None})


# ---------------------------------------------------------------------------
# 1. Hygiene & Connector Invariants
# ---------------------------------------------------------------------------
def test_importing_the_connector_does_not_load_requests():
    """Verify that importing bluesky does not eagerly import requests."""
    # Ensure requests was not imported as a side-effect of importing bluesky
    # (requests is imported on-demand inside BlueskyClient._get_session)
    assert hasattr(bluesky_module, "bluesky_source")
    assert hasattr(bluesky_module, "build_bluesky_service")


def test_client_token_hygiene():
    """Verify that credentials never leak into exception messages or string reprs."""
    secret_token = "SUPER_SECRET_BLUESKY_JWT_TOKEN_XYZ123"
    fake_session = FakeSession(
        [FakeResponse({"error": "AuthRequired"}, status_code=401)]
    )
    client = BlueskyClient(access_token=secret_token, session=fake_session)

    with pytest.raises(BlueskyAuthError) as exc_info:
        client.get_author_feed("alice.bsky.social")

    error_msg = str(exc_info.value)
    assert secret_token not in error_msg
    assert "SUPER_SECRET" not in error_msg
    assert "401" not in error_msg or "Authentication rejected" in error_msg


def test_client_rate_limiting_429():
    """Verify that HTTP 429 parses RateLimit-Reset and raises BlueskyRateLimitedError."""
    fake_session = FakeSession(
        [
            FakeResponse(
                {"error": "RateLimitExceeded"},
                status_code=429,
                headers={"RateLimit-Remaining": "0", "RateLimit-Reset": "1731540000"},
            )
        ]
    )
    client = BlueskyClient(session=fake_session)

    with pytest.raises(BlueskyRateLimitedError) as exc_info:
        client.get_author_feed("bsky.app")

    assert exc_info.value.reset_at_epoch == 1731540000
    assert client.rate_limit.get("remaining") == 0


def test_client_actor_not_found_404():
    """Verify that HTTP 404 raises BlueskyActorNotFoundError."""
    fake_session = FakeSession([FakeResponse({"error": "NotFound"}, status_code=404)])
    client = BlueskyClient(session=fake_session)

    with pytest.raises(BlueskyActorNotFoundError):
        client.get_profile("nonexistent-actor-404.bsky.social")


# ---------------------------------------------------------------------------
# 2. Rich Text Facet Parsing (UTF-8 Byte Slice Invariants)
# ---------------------------------------------------------------------------
def test_resolve_facets_links_mentions_tags():
    """Test resolving standard links, mentions, and hashtags."""
    text = "Explore Cognee memory at https://cognee.ai with @alice for #graphrag"
    link_target = "https://cognee.ai"
    mention_target = "@alice"
    tag_target = "#graphrag"

    b_text = text.encode("utf-8")
    b_link_start = len(text[: text.index(link_target)].encode("utf-8"))
    b_link_end = b_link_start + len(link_target.encode("utf-8"))

    b_mention_start = len(text[: text.index(mention_target)].encode("utf-8"))
    b_mention_end = b_mention_start + len(mention_target.encode("utf-8"))

    b_tag_start = len(text[: text.index(tag_target)].encode("utf-8"))
    b_tag_end = b_tag_start + len(tag_target.encode("utf-8"))

    facets = [
        {
            "index": {"byteStart": b_link_start, "byteEnd": b_link_end},
            "features": [
                {"$type": "app.bsky.richtext.facet#link", "uri": "https://cognee.ai"}
            ],
        },
        {
            "index": {"byteStart": b_mention_start, "byteEnd": b_mention_end},
            "features": [
                {"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:alice123"}
            ],
        },
        {
            "index": {"byteStart": b_tag_start, "byteEnd": b_tag_end},
            "features": [{"$type": "app.bsky.richtext.facet#tag", "tag": "graphrag"}],
        },
    ]

    resolved = resolve_facets(text, facets)
    assert "[https://cognee.ai](https://cognee.ai)" in resolved
    assert "[@alice](https://bsky.app/profile/did:plc:alice123)" in resolved
    assert "[#graphrag](https://bsky.app/hashtag/graphrag)" in resolved


def test_resolve_facets_with_multibyte_unicode_and_emojis():
    """Verify that 4-byte emojis do not cause byte-offset misalignment."""
    # 🚀 is 4 bytes in UTF-8: \xf0\x9f\x9a\x80
    # 🤖 is 4 bytes in UTF-8: \xf0\x9f\xa4\x96
    text = "🚀 AI Agents 🤖 test link https://example.com done!"
    link_sub = "https://example.com"
    b_text = text.encode("utf-8")
    b_start = len(text[: text.index(link_sub)].encode("utf-8"))
    b_end = b_start + len(link_sub.encode("utf-8"))

    facets = [
        {
            "index": {"byteStart": b_start, "byteEnd": b_end},
            "features": [
                {"$type": "app.bsky.richtext.facet#link", "uri": "https://example.com"}
            ],
        }
    ]

    resolved = resolve_facets(text, facets)
    assert "[https://example.com](https://example.com)" in resolved
    assert resolved.startswith("🚀 AI Agents 🤖 test link")
    assert resolved.endswith("done!")


def test_resolve_facets_handles_empty_none_and_malformed_indices():
    """Verify robustness against None, empty facets, and invalid boundary offsets."""
    assert resolve_facets("", None) == ""
    assert resolve_facets("Plain text", None) == "Plain text"
    assert resolve_facets("Plain text", []) == "Plain text"

    # Out of bounds facet offsets should be cleanly ignored without crash
    malformed_facets = [
        {"index": {"byteStart": 999, "byteEnd": 1050}, "features": []},
        {"index": {"byteStart": -5, "byteEnd": 10}, "features": []},
        {"index": {"byteStart": 20, "byteEnd": 5}, "features": []},  # inverted
    ]
    assert resolve_facets("Hello World", malformed_facets) == "Hello World"


# ---------------------------------------------------------------------------
# 3. Document Rendering & Context Flattening
# ---------------------------------------------------------------------------
def test_render_post_document_flat_schema():
    """Verify output dictionary structure against Cognee flat document schema."""
    feed_item = {
        "post": {
            "uri": "at://did:plc:test1234/app.bsky.feed.post/3n25abc",
            "author": {"handle": "alice.bsky.social", "displayName": "Alice"},
            "record": {
                "text": "Hello Cognee! Testing long-term agent memory integration.",
                "createdAt": "2026-11-13T10:00:00.000Z",
            },
        }
    }

    doc = render_post_document(feed_item)

    assert doc["id"] == "post:at://did:plc:test1234/app.bsky.feed.post/3n25abc"
    assert doc["title"].startswith("@alice.bsky.social: Hello Cognee!")
    assert doc["content"] == "Hello Cognee! Testing long-term agent memory integration."
    assert doc["url"] == "https://bsky.app/profile/alice.bsky.social/post/3n25abc"
    assert doc["_deleted"] is False


def test_render_post_document_with_reply_and_quote_flattening():
    """Verify that replies and quoted posts are flattened into content."""
    feed_item = {
        "post": {
            "uri": "at://did:plc:bob123/app.bsky.feed.post/3n25xyz",
            "author": {"handle": "bob.bsky.social", "displayName": "Bob"},
            "record": {
                "text": "I agree with this completely.",
                "createdAt": "2026-11-13T11:00:00.000Z",
            },
            "embed": {
                "$type": "app.bsky.embed.record#view",
                "record": {
                    "author": {"handle": "dan.bsky.social"},
                    "value": {
                        "text": "Decentralized memory layers are critical for AI."
                    },
                },
            },
        },
        "reply": {
            "parent": {
                "author": {"handle": "carol.bsky.social"},
                "record": {"text": "What do you think about agent memory?"},
            }
        },
    }

    doc = render_post_document(feed_item)

    # Check that parent reply is folded in
    assert (
        "> Replying to @carol.bsky.social: What do you think about agent memory?"
        in doc["content"]
    )
    assert "I agree with this completely." in doc["content"]
    # Check that quoted post is folded in
    assert (
        "> Quoting @dan.bsky.social: Decentralized memory layers are critical for AI."
        in doc["content"]
    )


def test_extract_post_web_url():
    """Verify URL conversion from AT-URI to bsky.app web URL."""
    uri = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.post/3n25midjy722x"
    url = extract_post_web_url("bsky.app", uri)
    assert url == "https://bsky.app/profile/bsky.app/post/3n25midjy722x"


# ---------------------------------------------------------------------------
# 4. Incremental Sync & State Invariants
# ---------------------------------------------------------------------------
def test_incremental_sync_second_run_yields_zero():
    """Verify that a second run with the same posts yields strictly 0 new documents."""
    posts_page_1 = [
        {
            "post": {
                "uri": "at://did:plc:test1/app.bsky.feed.post/p1",
                "author": {"handle": "alice.bsky.social"},
                "record": {"text": "Post 1", "createdAt": "2026-11-13T10:00:00.000Z"},
            }
        },
        {
            "post": {
                "uri": "at://did:plc:test1/app.bsky.feed.post/p2",
                "author": {"handle": "alice.bsky.social"},
                "record": {"text": "Post 2", "createdAt": "2026-11-13T10:05:00.000Z"},
            }
        },
    ]

    session_1 = FakeSession([FakeResponse({"feed": posts_page_1, "cursor": None})])
    client_1 = BlueskyClient(session=session_1)

    resource_state: dict[str, Any] = {}
    stats_1: dict[str, int] = {}

    # Run 1: Should yield 2 documents
    run_1_docs = list(
        _iter_rows(client_1, "alice.bsky.social", resource_state, stats_1)
    )
    assert len(run_1_docs) == 2
    assert "post:at://did:plc:test1/app.bsky.feed.post/p1" in resource_state["seen_ids"]
    assert "post:at://did:plc:test1/app.bsky.feed.post/p2" in resource_state["seen_ids"]
    assert resource_state["floor_timestamp"] == "2026-11-13T10:05:00.000Z"

    # Run 2: Same feed upstream — MUST yield 0 new documents
    session_2 = FakeSession([FakeResponse({"feed": posts_page_1, "cursor": None})])
    client_2 = BlueskyClient(session=session_2)
    stats_2: dict[str, int] = {}

    run_2_docs = list(
        _iter_rows(client_2, "alice.bsky.social", resource_state, stats_2)
    )
    assert len(run_2_docs) == 0
    assert stats_2.get("skipped", 0) == 2


def test_sync_halts_cleanly_on_rate_limit_with_state_preserved():
    """Verify that when 429 occurs, sync halts with state saved and stats recorded."""
    posts_page_1 = [
        {
            "post": {
                "uri": "at://did:plc:test1/app.bsky.feed.post/p1",
                "author": {"handle": "alice.bsky.social"},
                "record": {"text": "Post 1", "createdAt": "2026-11-13T09:00:00.000Z"},
            }
        }
    ]

    # Page 1 succeeds, Page 2 hits 429
    session = FakeSession(
        [
            FakeResponse({"feed": posts_page_1, "cursor": "page_2_cursor"}),
            FakeResponse(
                {"error": "RateLimit"},
                status_code=429,
                headers={"RateLimit-Reset": "1731550000"},
            ),
        ]
    )
    client = BlueskyClient(session=session)

    resource_state: dict[str, Any] = {}
    stats: dict[str, int] = {}

    docs = list(
        _iter_rows(client, "alice.bsky.social", resource_state, stats, max_requests=5)
    )

    # Page 1 docs were yielded
    assert len(docs) == 1
    # Rate limit was flagged in stats
    assert stats.get("failed_rate_limit") == 1
    # State from page 1 was preserved
    assert "post:at://did:plc:test1/app.bsky.feed.post/p1" in resource_state["seen_ids"]


def test_source_resource_attributes():
    """Verify DLT source metadata attributes and merge configuration."""
    fake_session = FakeSession([FakeResponse({"feed": [], "cursor": None})])
    client = BlueskyClient(session=fake_session)

    source = bluesky_source(actor="alice.bsky.social", service=client)

    assert getattr(source, "name") == "bluesky_alice_bsky_social"
    assert getattr(source, "write_disposition") == "merge"
    assert getattr(source, bluesky_module.DOCUMENT_SOURCE_ATTR, None) == "bluesky"
    assert hasattr(source, "cognee_sync_stats")


# ---------------------------------------------------------------------------
# 5. Live Public XRPC Integration Smoke Test
# ---------------------------------------------------------------------------
def test_live_public_xrpc_smoke():
    """Test against real public Bluesky API without any credentials."""
    client = BlueskyClient()  # Uses live PUBLIC_XRPC_BASE
    feed_data = client.get_author_feed("bsky.app", limit=3)

    assert "feed" in feed_data
    feed = feed_data["feed"]
    assert len(feed) > 0

    first_item = feed[0]
    doc = render_post_document(first_item)

    assert doc["id"].startswith("post:at://")
    assert doc["url"].startswith("https://bsky.app/profile/")
    assert doc["_deleted"] is False
    assert len(doc["content"]) > 0
