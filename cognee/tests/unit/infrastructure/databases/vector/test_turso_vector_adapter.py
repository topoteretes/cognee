import sys
import types
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from cognee.infrastructure.databases.vector.config import VectorConfig


class TestVectorConfigTursoHandler:
    """Verify that VectorConfig auto-sets the dataset handler for turso."""

    def test_turso_provider_sets_turso_handler(self):
        """When VECTOR_DB_PROVIDER=turso, handler should auto-switch to 'turso'."""
        with patch.dict(
            "os.environ",
            {"VECTOR_DB_PROVIDER": "turso"},
            clear=False,
        ):
            config = VectorConfig(
                vector_db_provider="turso",
                vector_db_url="libsql://test.turso.io",
            )
            assert config.vector_dataset_database_handler == "turso"

    def test_turso_provider_preserves_explicit_handler(self):
        """When handler is explicitly set to something other than lancedb/turso, keep it."""
        config = VectorConfig(
            vector_db_provider="turso",
            vector_dataset_database_handler="custom_handler",
            vector_db_url="libsql://test.turso.io",
        )
        assert config.vector_dataset_database_handler == "custom_handler"

    def test_lancedb_provider_unchanged(self):
        """Ensure lancedb provider still defaults correctly."""
        config = VectorConfig(
            vector_db_provider="lancedb",
        )
        assert config.vector_dataset_database_handler == "lancedb"


class TestTursoVectorFactoryBranch:
    """Verify the turso branch in create_vector_engine."""

    def test_turso_missing_deps_raises_import_error(self):
        """When sqlalchemy_libsql is not installed, ImportError should be raised."""
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )

        _create_vector_engine.cache_clear()

        with patch.dict(sys.modules, {"sqlalchemy_libsql": None}):
            with pytest.raises(ImportError, match=r"pip install cognee.*turso"):
                _create_vector_engine(
                    vector_db_provider="turso",
                    vector_db_url="libsql://test.turso.io",
                    vector_db_name="test_db",
                    vector_db_port="",
                    vector_db_key="test-token",
                    vector_dataset_database_handler="turso",
                    vector_db_username="",
                    vector_db_password="",
                    vector_db_host="",
                    vector_db_subprocess_enabled=True,
                )

        _create_vector_engine.cache_clear()

    def test_turso_factory_returns_adapter(self):
        """When sqlalchemy_libsql is available, factory should return TursoVectorAdapter."""
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )

        _create_vector_engine.cache_clear()

        # Inject fake sqlalchemy_libsql
        fake_libsql = types.ModuleType("sqlalchemy_libsql")
        fake_aiolibsql = types.ModuleType("sqlalchemy_libsql.aiolibsql")
        fake_libsql.aiolibsql = fake_aiolibsql

        mock_embedding_engine = MagicMock()
        mock_embedding_engine.get_vector_size.return_value = 384

        with patch.dict(
            sys.modules,
            {
                "sqlalchemy_libsql": fake_libsql,
                "sqlalchemy_libsql.aiolibsql": fake_aiolibsql,
            },
        ):
            with patch(
                "cognee.infrastructure.databases.vector.create_vector_engine.get_embedding_engine",
                return_value=mock_embedding_engine,
            ):
                with patch(
                    "cognee.infrastructure.databases.vector.turso.TursoVectorAdapter"
                    ".TursoVectorAdapter.__init__",
                    return_value=None,
                ):
                    result = _create_vector_engine(
                        vector_db_provider="turso",
                        vector_db_url="libsql://test.turso.io",
                        vector_db_name="test_db",
                        vector_db_port="",
                        vector_db_key="test-token",
                        vector_dataset_database_handler="turso",
                        vector_db_username="",
                        vector_db_password="",
                        vector_db_host="",
                        vector_db_subprocess_enabled=True,
                    )

                    from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
                        TursoVectorAdapter,
                    )

                    assert isinstance(result, TursoVectorAdapter)

        _create_vector_engine.cache_clear()


class TestSupportedDatasetDatabaseHandlers:
    """Verify turso is registered in the handler registry."""

    def test_turso_handler_registered(self):
        """Turso should be in supported_dataset_database_handlers."""
        from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
            supported_dataset_database_handlers,
        )

        assert "turso" in supported_dataset_database_handlers
        assert supported_dataset_database_handlers["turso"]["handler_provider"] == "turso"
