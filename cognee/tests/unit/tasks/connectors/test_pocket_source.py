"""Unit tests for the Pocket reading list data-source connector."""

import os
import sys
import unittest
from unittest.mock import MagicMock

try:
    from cognee.tasks.ingestion.connectors.pocket import (
        PocketAuthError,
        PocketClient,
        PocketError,
        PocketRateLimitError,
        format_pocket_document,
        pocket_source,
    )
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../tasks/ingestion/connectors")))
    from pocket import (
        PocketAuthError,
        PocketClient,
        PocketError,
        PocketRateLimitError,
        format_pocket_document,
        pocket_source,
    )


class TestPocketConnector(unittest.TestCase):
    """Test suite for Pocket reading list connector."""

    def test_pocket_client_validation_error(self):
        """Test that empty consumer key or access token raises ValueError."""
        with self.assertRaises(ValueError):
            PocketClient(consumer_key="", access_token="valid_token")
        with self.assertRaises(ValueError):
            PocketClient(consumer_key="valid_key", access_token="  ")

    def test_format_pocket_document_normal(self):
        """Test formatting a standard unread article."""
        item = {
            "item_id": "123456",
            "resolved_title": "Understanding Large Language Models",
            "resolved_url": "https://example.com/llm-guide",
            "excerpt": "A detailed technical guide explaining transformer mechanics.",
            "status": "0",
            "time_added": "1731150000",
        }
        doc = format_pocket_document(item)
        self.assertEqual(doc["id"], "pocket:item:123456")
        self.assertEqual(doc["title"], "Pocket: Understanding Large Language Models")
        self.assertIn("Understanding Large Language Models", doc["content"])
        self.assertIn("A detailed technical guide", doc["content"])
        self.assertEqual(doc["_deleted"], False)
        self.assertEqual(doc["cognee_node_set"], "pocket_reading_list")
        self.assertEqual(doc["url"], "https://example.com/llm-guide")

    def test_format_pocket_document_deleted_tombstone(self):
        """Test formatting a deleted article emits _deleted: True."""
        item = {
            "item_id": "999888",
            "resolved_title": "Old Discarded Draft",
            "status": "2",  # Pocket status 2 indicates deleted
        }
        doc = format_pocket_document(item)
        self.assertEqual(doc["id"], "pocket:item:999888")
        self.assertEqual(doc["_deleted"], True)

    def test_format_pocket_document_authors_and_tags(self):
        """Test metadata extraction for authors and user tags."""
        item = {
            "item_id": "555123",
            "resolved_title": "Autonomous Agent Design",
            "resolved_url": "https://example.com/agent-design",
            "authors": {"1": {"name": "Dr. Sarah Chen"}, "2": {"name": "Alex Mercer"}},
            "tags": {"ai": {}, "agents": {}, "production": {}},
            "status": "0",
        }
        doc = format_pocket_document(item)
        self.assertIn("Dr. Sarah Chen, Alex Mercer", doc["content"])
        self.assertIn("ai, agents, production", doc["content"])
        self.assertEqual(doc["_deleted"], False)

    def test_pocket_source_yields_documents(self):
        """Test yielding articles from the pocket_source generator."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": 1,
            "list": {
                "101": {
                    "item_id": "101",
                    "resolved_title": "Article One",
                    "resolved_url": "https://example.com/1",
                    "status": "0",
                },
                "102": {
                    "item_id": "102",
                    "resolved_title": "Article Two",
                    "resolved_url": "https://example.com/2",
                    "status": "0",
                },
            },
        }
        mock_session.post.return_value = mock_resp

        results = list(
            pocket_source(
                consumer_key="test_consumer",
                access_token="test_token",
                limit=10,
                session=mock_session,
            )
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "pocket:item:101")
        self.assertEqual(results[1]["id"], "pocket:item:102")

    def test_pocket_rate_limit_error(self):
        """Test that HTTP 429 raises PocketRateLimitError."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_session.post.return_value = mock_resp

        client = PocketClient("ckey", "atoken", session=mock_session)
        with self.assertRaises(PocketRateLimitError):
            client.get_items()

    def test_pocket_auth_error_401(self):
        """Test that HTTP 401 raises PocketAuthError."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_session.post.return_value = mock_resp

        client = PocketClient("ckey", "bad_token", session=mock_session)
        with self.assertRaises(PocketAuthError):
            client.get_items()


if __name__ == "__main__":
    unittest.main()
