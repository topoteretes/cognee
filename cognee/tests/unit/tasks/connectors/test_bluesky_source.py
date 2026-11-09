"""Unit tests for the Bluesky (AT Protocol) data-source connector."""

import os
import sys
import unittest
from unittest.mock import MagicMock

try:
    from cognee.tasks.ingestion.connectors.bluesky import (
        BlueskyAuthError,
        BlueskyClient,
        BlueskyRateLimitError,
        bluesky_source,
        format_post_document,
        resolve_facets,
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tasks/ingestion/connectors")))
    from bluesky import (
        BlueskyAuthError,
        BlueskyClient,
        BlueskyRateLimitError,
        bluesky_source,
        format_post_document,
        resolve_facets,
    )


class TestBlueskyConnector(unittest.TestCase):
    """Test suite for Bluesky data-source connector."""

    def test_resolve_facets(self):
        """Test resolving rich text facets (links and mentions) to readable markdown."""
        text = "Hello @alice check https://example.com today"
        facets = [
            {
                "index": {"byteStart": 6, "byteEnd": 12},
                "features": [{"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:alice123"}],
            },
            {
                "index": {"byteStart": 19, "byteEnd": 38},
                "features": [{"$type": "app.bsky.richtext.facet#link", "uri": "https://example.com"}],
            },
        ]
        resolved = resolve_facets(text, facets)
        self.assertIn("@alice", resolved)
        self.assertIn("[https://example.com](https://example.com)", resolved)

    def test_format_post_document(self):
        """Test document formatting with proper metadata and node set."""
        post = {
            "uri": "at://did:plc:test1234/app.bsky.feed.post/3kabc123",
            "author": {
                "handle": "alice.bsky.social",
                "displayName": "Alice Engineer",
            },
            "record": {
                "text": "Launching new features for AI memory!",
                "createdAt": "2026-11-09T10:00:00.000Z",
            },
        }
        doc = format_post_document(post, reply_context="In reply to @bob: Great roadmap")
        self.assertEqual(doc["id"], "bluesky:post:at://did:plc:test1234/app.bsky.feed.post/3kabc123")
        self.assertIn("Alice Engineer (@alice.bsky.social)", doc["content"])
        self.assertIn("In reply to @bob: Great roadmap", doc["content"])
        self.assertIn("Launching new features for AI memory!", doc["content"])
        self.assertEqual(doc["_deleted"], False)
        self.assertEqual(doc["cognee_node_set"], "bluesky_posts")
        self.assertEqual(doc["url"], "https://bsky.app/profile/alice.bsky.social/post/3kabc123")

    def test_bluesky_source_author_feed(self):
        """Test yielding posts from an author feed."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "feed": [
                {
                    "post": {
                        "uri": "at://did:plc:test/app.bsky.feed.post/1",
                        "author": {"handle": "tester.bsky.social"},
                        "record": {"text": "Test post 1", "createdAt": "2026-11-09T08:00:00Z"},
                    }
                },
                {
                    "post": {
                        "uri": "at://did:plc:test/app.bsky.feed.post/2",
                        "author": {"handle": "tester.bsky.social"},
                        "record": {"text": "Test post 2", "createdAt": "2026-11-09T09:00:00Z"},
                    }
                },
            ],
            "cursor": None,
        }
        mock_session.get.return_value = mock_resp

        results = list(bluesky_source(handle="tester.bsky.social", limit=10, session=mock_session))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "bluesky:post:at://did:plc:test/app.bsky.feed.post/1")
        self.assertEqual(results[1]["id"], "bluesky:post:at://did:plc:test/app.bsky.feed.post/2")

    def test_bluesky_rate_limit(self):
        """Test that rate limit responses trigger BlueskyRateLimitError."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_session.get.return_value = mock_resp

        client = BlueskyClient(session=mock_session)
        with self.assertRaises(BlueskyRateLimitError):
            client.get_author_feed("busy.user")

    def test_bluesky_auth_failure(self):
        """Test authentication error handling on invalid credentials."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_session.post.return_value = mock_resp

        with self.assertRaises(BlueskyAuthError):
            BlueskyClient(identifier="invalid.user", app_password="badpassword", session=mock_session)

    def test_bluesky_source_search(self):
        """Test yielding posts from search queries."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "posts": [
                {
                    "uri": "at://did:plc:search/app.bsky.feed.post/10",
                    "author": {"handle": "agent.bsky.social"},
                    "record": {"text": "Agent search result", "createdAt": "2026-11-09T10:00:00Z"},
                }
            ],
            "cursor": None,
        }
        mock_session.get.return_value = mock_resp

        results = list(bluesky_source(query="agent memory", limit=5, session=mock_session))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "bluesky:post:at://did:plc:search/app.bsky.feed.post/10")
        self.assertIn("Agent search result", results[0]["content"])

    def test_format_post_document_with_quote_embed(self):
        """Test embedding quoted post content."""
        post = {
            "uri": "at://did:plc:quote/app.bsky.feed.post/99",
            "author": {"handle": "commenter.bsky.social", "displayName": "Commenter"},
            "record": {"text": "Look at this announcement!", "createdAt": "2026-11-09T10:00:00Z"},
            "embed": {
                "$type": "app.bsky.embed.record#view",
                "record": {
                    "$type": "app.bsky.feed.defs#postView",
                    "author": {"handle": "original.bsky.social"},
                    "value": {"text": "Original announcement text"},
                },
            },
        }
        doc = format_post_document(post)
        self.assertIn("Look at this announcement!", doc["content"])
        self.assertIn("Quoted Post from @original.bsky.social", doc["content"])
        self.assertIn("Original announcement text", doc["content"])


if __name__ == "__main__":
    unittest.main()
