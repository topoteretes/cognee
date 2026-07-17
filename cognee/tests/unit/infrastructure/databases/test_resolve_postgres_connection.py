"""Unit tests for the shared Postgres connection resolver."""

import pytest
from sqlalchemy.engine import make_url

from cognee.infrastructure.databases.utils.resolve_postgres_connection import (
    resolve_postgres_url,
)


# Shared DB_* block used as the fallback group across tests.
SHARED = {
    "shared_host": "shared-host",
    "shared_port": "5432",
    "shared_username": "shared_user",
    "shared_password": "shared_pass",
    "shared_name": "shared_db",
}

# A complete store-level credential group.
STORE = {
    "store_host": "store-host",
    "store_port": "6543",
    "store_username": "store_user",
    "store_password": "store_pass",
    "store_name": "store_db",
}

MISSING_ERROR = "Missing required credentials."


class TestResolvePostgresUrl:
    """Precedence: URL wins -> store group -> shared DB_* group -> error."""

    def test_url_wins_over_parts(self):
        """A store URL is authoritative even when discrete parts are also set."""
        url = resolve_postgres_url(
            store_url="postgresql+asyncpg://url_user:url_pass@url-host:1111/url_db",
            **STORE,
            **SHARED,
            missing_error=MISSING_ERROR,
        )

        assert url == make_url("postgresql+asyncpg://url_user:url_pass@url-host:1111/url_db")
        # The discrete store / shared parts must NOT leak into the result.
        assert url.host == "url-host"
        assert url.port == 1111
        assert url.username == "url_user"
        assert url.database == "url_db"

    def test_store_parts_used_when_store_group_complete(self):
        """When the store supplies its own credentials, the store group wins."""
        url = resolve_postgres_url(
            store_url="",
            **STORE,
            **SHARED,
            missing_error=MISSING_ERROR,
        )

        assert url.drivername == "postgresql+asyncpg"
        assert url.host == "store-host"
        assert url.port == 6543  # store's own port travels with the store group
        assert url.username == "store_user"
        assert url.password == "store_pass"
        assert url.database == "store_db"

    def test_shared_group_used_with_its_own_port(self):
        """Store core creds absent -> the entire shared DB_* group is used,
        including the shared port. The store's (placeholder) port must NOT be
        mixed into the shared group."""
        url = resolve_postgres_url(
            store_url="",
            store_host="",
            store_port="1234",  # placeholder default; must be ignored, not mixed in
            store_username="",
            store_password="",
            store_name="",
            **SHARED,
            missing_error=MISSING_ERROR,
        )

        assert url.host == "shared-host"
        assert url.port == 5432  # shared port, not the store placeholder 1234
        assert url.username == "shared_user"
        assert url.password == "shared_pass"
        assert url.database == "shared_db"

    def test_special_characters_in_credentials_are_escaped(self):
        """Usernames/passwords with #/@/: must round-trip via URL.create."""
        url = resolve_postgres_url(
            store_url="",
            store_host="store-host",
            store_port="6543",
            store_username="user#name",
            store_password="p@ss:word",
            store_name="store_db",
            **SHARED,
            missing_error=MISSING_ERROR,
        )

        assert url.username == "user#name"
        assert url.password == "p@ss:word"
        # Rendered form must escape the special characters.
        rendered = url.render_as_string(hide_password=False)
        assert "user%23name" in rendered
        assert "p%40ss%3Aword" in rendered

    def test_incomplete_groups_raise_environment_error(self):
        """Neither the store nor the shared group is complete -> raise."""
        with pytest.raises(EnvironmentError, match=MISSING_ERROR):
            resolve_postgres_url(
                store_url="",
                store_host="",
                store_port="1234",
                store_username="",
                store_password="",
                store_name="",
                shared_host="",
                shared_port="",
                shared_username="",
                shared_password="",
                shared_name="",
                missing_error=MISSING_ERROR,
            )
