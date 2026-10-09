import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)

POSTGRES_PARAMS = {
    "db_path": "/tmp",
    "db_name": "test_db",
    "db_host": "localhost",
    "db_port": "5432",
    "db_username": "user",
    "db_password": "pass",
    "db_provider": "postgres",
}

TURSO_PARAMS = {
    "db_path": "/tmp",
    "db_name": "test_db",
    "db_host": "",
    "db_port": "",
    "db_username": "",
    "db_password": "",
    "db_provider": "turso",
}


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """create_relational_engine is wrapped with @lru_cache — clear between tests."""
    create_relational_engine.cache_clear()
    yield
    create_relational_engine.cache_clear()


@pytest.fixture(autouse=True)
def _fake_asyncpg():
    """Inject a fake asyncpg module so the import inside the function succeeds."""
    fake = types.ModuleType("asyncpg")
    with patch.dict(sys.modules, {"asyncpg": fake}):
        yield


class TestCreateRelationalEngineSpecialCharacters:
    """Verify that special characters in credentials are handled correctly by URL.create."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_postgres_special_chars_in_username_and_password(self, mock_adapter):
        """Username and password with special characters should round-trip correctly."""
        create_relational_engine(
            **{**POSTGRES_PARAMS, "db_username": "user#name", "db_password": "p@ss:word"}
        )

        url = mock_adapter.call_args[0][0]
        assert url.username == "user#name"
        assert url.password == "p@ss:word"


class TestCreateRelationalEngineConnectArgs:
    """Verify that connect_args are forwarded to the SQLAlchemyAdapter."""

    @patch("cognee.infrastructure.databases.relational.create_relational_engine.SQLAlchemyAdapter")
    def test_postgres_no_connect_args_passes_empty_dict(self, mock_adapter):
        """When no connect_args are provided, an empty dict should be forwarded to the adapter."""
        create_relational_engine(**POSTGRES_PARAMS)

        _, kwargs = mock_adapter.call_args
        assert kwargs.get("connect_args") == {}


class TestCreateRelationalEngineTurso:
    """Verify the DB_PROVIDER=turso branch hands the local file path to TursoAdapter."""

    @pytest.fixture(autouse=True)
    def _fake_turso(self):
        """Inject a fake ``turso`` module so the driver probe inside the branch succeeds."""
        fake = types.ModuleType("turso")
        with patch.dict(sys.modules, {"turso": fake}):
            yield

    @patch("cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter.TursoAdapter")
    def test_turso_returns_turso_adapter_with_local_path(self, mock_turso_adapter):
        """turso provider passes ``db_path/db_name`` and the connection settings to TursoAdapter."""
        engine = create_relational_engine(
            **TURSO_PARAMS,
            database_connect_args=(("check_same_thread", False),),
            pool_args=(("poolclass", "nullpool"),),
        )

        assert engine is mock_turso_adapter.return_value
        args, kwargs = mock_turso_adapter.call_args
        assert args[0] == f"{TURSO_PARAMS['db_path']}/{TURSO_PARAMS['db_name']}"
        assert kwargs["connect_args"] == {"check_same_thread": False}
        assert kwargs["pool_args"] == {"poolclass": "nullpool"}

    @pytest.mark.parametrize(
        "remote",
        [
            {"db_turso_url": "libsql://db.turso.io"},
            {"db_turso_auth_token": "tok"},
            {"db_turso_url": "libsql://db.turso.io", "db_turso_auth_token": "tok"},
        ],
    )
    @patch("cognee.infrastructure.databases.relational.sqlalchemy.TursoAdapter.TursoAdapter")
    def test_turso_remote_settings_are_rejected(self, mock_turso_adapter, remote):
        """Remote Turso is out of scope for the local backend: fail loudly, never fall back."""
        with pytest.raises(OSError, match="Remote Turso"):
            create_relational_engine(**{**TURSO_PARAMS, **remote})

        mock_turso_adapter.assert_not_called()

    def test_turso_missing_driver_raises_actionable_import_error(self):
        """When pyturso is missing, raise a clear cognee[turso] install error."""
        # Setting the module to None in sys.modules makes `import turso` raise ImportError.
        with (
            patch.dict(sys.modules, {"turso": None}),
            pytest.raises(ImportError, match="Turso dependencies"),
        ):
            create_relational_engine(**TURSO_PARAMS)
