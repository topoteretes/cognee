import os
import pytest
from unittest.mock import patch
from cognee.infrastructure.databases.relational.config import RelationalConfig
from cognee.infrastructure.databases.vector.config import VectorConfig
from cognee.infrastructure.databases.graph.config import GraphConfig


SHARED_PG = {
    "DB_HOST": "shared-host",
    "DB_PORT": "5432",
    "DB_USERNAME": "shared-user",
    "DB_PASSWORD": "shared-pass",
    "DB_NAME": "shared-db",
    "DB_PROVIDER": "postgres",
}


EMPTY_GRAPH = {
    "GRAPH_DATABASE_HOST": "",
    "GRAPH_DATABASE_PORT": "",
    "GRAPH_DATABASE_USERNAME": "",
    "GRAPH_DATABASE_PASSWORD": "",
    "GRAPH_DATABASE_NAME": "",
}


class TestProviderAlias:
    """VECTOR_DB_PROVIDER=postgres must be normalised to pgvector."""

    def test_postgres_alias_normalised_to_pgvector(self):
        """VECTOR_DB_PROVIDER=postgres should be stored as pgvector."""
        with patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "postgres"}):
            cfg = VectorConfig()
            assert cfg.vector_db_provider == "pgvector"

    def test_pgvector_unchanged(self):
        """VECTOR_DB_PROVIDER=pgvector should remain pgvector."""
        with patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "pgvector"}):
            cfg = VectorConfig()
            assert cfg.vector_db_provider == "pgvector"

    def test_other_provider_unchanged(self):
        """Non-postgres providers must not be affected."""
        with patch.dict(os.environ, {"VECTOR_DB_PROVIDER": "lancedb"}):
            cfg = VectorConfig()
            assert cfg.vector_db_provider == "lancedb"


class TestRelationalDbUrl:
    """DB_URL support for relational config."""

    def test_db_url_stored_on_config(self):
        """DB_URL should be stored on RelationalConfig."""
        url = "postgresql+asyncpg://user:pass@host:5432/mydb"
        with patch.dict(os.environ, {"DB_URL": url}):
            cfg = RelationalConfig()
            assert cfg.db_url == url

    def test_db_url_empty_by_default(self):
        """db_url should default to empty string when not set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = RelationalConfig()
            assert cfg.db_url == ""

    def test_db_url_in_to_dict(self):
        """db_url must appear in to_dict() so the engine factory receives it."""
        url = "postgresql+asyncpg://user:pass@host:5432/mydb"
        with patch.dict(os.environ, {"DB_URL": url}):
            cfg = RelationalConfig()
            assert "db_url" in cfg.to_dict()
            assert cfg.to_dict()["db_url"] == url


class TestSharedCredentialFallback:
    """DB_* must be the shared credential block for all-Postgres deployments."""

    def test_db_star_alone_sets_relational_config(self):
        """DB_* fields should populate RelationalConfig correctly."""
        with patch.dict(os.environ, SHARED_PG):
            cfg = RelationalConfig()
            assert cfg.db_host == "shared-host"
            assert cfg.db_port == "5432"
            assert cfg.db_username == "shared-user"
            assert cfg.db_password == "shared-pass"
            assert cfg.db_name == "shared-db"

    def test_vector_specific_overrides_shared(self):
        """Fully set VECTOR_DB_* must take priority over DB_*."""
        env = {
            **SHARED_PG,
            "VECTOR_DB_PROVIDER": "pgvector",
            "VECTOR_DB_HOST": "vector-host",
            "VECTOR_DB_PORT": "5433",
            "VECTOR_DB_USERNAME": "vector-user",
            "VECTOR_DB_PASSWORD": "vector-pass",
            "VECTOR_DB_NAME": "vector-db",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = VectorConfig()
            assert cfg.vector_db_host == "vector-host"
            assert cfg.vector_db_username == "vector-user"

    def test_graph_specific_overrides_shared(self):
        """Fully set GRAPH_DATABASE_* must take priority over DB_*."""
        env = {
            **SHARED_PG,
            **EMPTY_GRAPH,
            "GRAPH_DATABASE_PROVIDER": "postgres",
            "GRAPH_DATABASE_HOST": "graph-host",
            "GRAPH_DATABASE_PORT": "5434",
            "GRAPH_DATABASE_USERNAME": "graph-user",
            "GRAPH_DATABASE_PASSWORD": "graph-pass",
            "GRAPH_DATABASE_NAME": "graph-db",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = GraphConfig()
            assert cfg.graph_database_host == "graph-host"
            assert cfg.graph_database_username == "graph-user"

    def test_partial_vector_override_does_not_mix(self):
        """
        When only VECTOR_DB_HOST is set (partial override), the config
        stores it as-is — the engine factory will fall back entirely to
        DB_* because not all five fields are present.
        This test locks down the all-or-nothing semantics.
        """
        # Only set HOST — omit PORT entirely (vector_db_port is int, rejects empty string)
        env = {
            **SHARED_PG,
            "VECTOR_DB_PROVIDER": "pgvector",
            "VECTOR_DB_HOST": "vector-host",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = VectorConfig()
            # Host is stored, but username/password/name are empty — factory will fall back
            assert cfg.vector_db_host == "vector-host"
            assert not cfg.vector_db_username


class TestMixedProviderUnaffected:
    """Non-Postgres providers must not be affected by DB_* fallback."""

    def test_lancedb_unaffected(self):
        """LanceDB provider should not be changed by DB_* being set."""
        env = {**SHARED_PG, "VECTOR_DB_PROVIDER": "lancedb"}
        with patch.dict(os.environ, env, clear=True):
            cfg = VectorConfig()
            assert cfg.vector_db_provider == "lancedb"

    def test_neo4j_graph_unaffected(self):
        """Neo4j graph provider should not be changed by DB_* being set."""
        env = {
            **SHARED_PG,
            "GRAPH_DATABASE_PROVIDER": "neo4j",
            "GRAPH_DATABASE_URL": "bolt://localhost:7687",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = GraphConfig()
            assert cfg.graph_database_provider == "neo4j"


class TestDbUrlFallback:
    """DB_URL alone must work — no crash when broken-out fields are absent."""

    def test_db_url_in_relational_config_when_no_broken_out_fields(self):
        """
        Setting only DB_URL (no DB_HOST/PORT/etc.) must populate db_url on
        RelationalConfig so the engine factory can use it directly.
        """
        url = "postgresql+asyncpg://user:pass@host:5432/mydb"
        env = {
            "DB_URL": url,
            "DB_PROVIDER": "postgres",
            "DB_HOST": "",
            "DB_PORT": "",
            "DB_USERNAME": "",
            "DB_PASSWORD": "",
            "DB_NAME": "",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = RelationalConfig()
            assert cfg.db_url == url
            assert not cfg.db_host
