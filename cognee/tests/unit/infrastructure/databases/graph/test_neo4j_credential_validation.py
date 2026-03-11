"""Tests for Neo4j adapter credential validation.

Verifies that the adapter fails fast with clear error messages when
credentials are incomplete, and requires explicit opt-in for anonymous access.

Related issue: https://github.com/topoteretes/cognee/issues/2307
"""

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


class TestResolveAuth:
    """Tests for Neo4jAdapter._resolve_auth static method."""

    def test_both_credentials_provided_returns_auth_tuple(self):
        result = Neo4jAdapter._resolve_auth("neo4j", "password123", allow_anonymous=False)
        assert result == ("neo4j", "password123")

    def test_username_only_raises_value_error(self):
        with pytest.raises(ValueError, match="GRAPH_DATABASE_PASSWORD is missing"):
            Neo4jAdapter._resolve_auth("neo4j", None, allow_anonymous=False)

    def test_password_only_raises_value_error(self):
        with pytest.raises(ValueError, match="GRAPH_DATABASE_USERNAME is missing"):
            Neo4jAdapter._resolve_auth(None, "password123", allow_anonymous=False)

    def test_username_only_with_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="GRAPH_DATABASE_PASSWORD is missing"):
            Neo4jAdapter._resolve_auth("neo4j", "", allow_anonymous=False)

    def test_password_only_with_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="GRAPH_DATABASE_USERNAME is missing"):
            Neo4jAdapter._resolve_auth("", "password123", allow_anonymous=False)

    def test_no_credentials_without_anonymous_raises_value_error(self):
        with pytest.raises(ValueError, match="Neo4j credentials not provided"):
            Neo4jAdapter._resolve_auth(None, None, allow_anonymous=False)

    def test_empty_strings_without_anonymous_raises_value_error(self):
        with pytest.raises(ValueError, match="Neo4j credentials not provided"):
            Neo4jAdapter._resolve_auth("", "", allow_anonymous=False)

    def test_no_credentials_with_anonymous_returns_none(self):
        result = Neo4jAdapter._resolve_auth(None, None, allow_anonymous=True)
        assert result is None

    def test_empty_strings_with_anonymous_returns_none(self):
        result = Neo4jAdapter._resolve_auth("", "", allow_anonymous=True)
        assert result is None

    def test_incomplete_credentials_ignore_anonymous_flag(self):
        """Even with allow_anonymous=True, incomplete credentials should raise."""
        with pytest.raises(ValueError, match="credentials incomplete"):
            Neo4jAdapter._resolve_auth("neo4j", None, allow_anonymous=True)

    def test_incomplete_credentials_error_suggests_env_vars(self):
        """Error message should tell the user exactly which env vars to set."""
        with pytest.raises(ValueError, match="GRAPH_DATABASE_ALLOW_ANONYMOUS"):
            Neo4jAdapter._resolve_auth("neo4j", None, allow_anonymous=False)

    def test_both_credentials_with_anonymous_flag_still_uses_auth(self):
        """When credentials are provided, allow_anonymous is irrelevant."""
        result = Neo4jAdapter._resolve_auth("neo4j", "pass", allow_anonymous=True)
        assert result == ("neo4j", "pass")


class TestDriverInjection:
    """Tests that credential validation is skipped when a driver is injected."""

    def test_injected_driver_skips_credential_validation(self):
        """When a pre-configured driver is passed, no credentials are needed."""
        mock_driver = object()
        adapter = Neo4jAdapter(
            graph_database_url="bolt://localhost:7687",
            driver=mock_driver,
        )
        assert adapter.driver is mock_driver

    def test_injected_driver_with_no_credentials_does_not_raise(self):
        """Injecting a driver with missing credentials should not fail."""
        mock_driver = object()
        # This would raise ValueError without the driver, but should succeed with it
        adapter = Neo4jAdapter(
            graph_database_url="bolt://localhost:7687",
            graph_database_username="neo4j",
            graph_database_password=None,
            driver=mock_driver,
        )
        assert adapter.driver is mock_driver
