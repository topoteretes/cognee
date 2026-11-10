"""A SQLite connection string has no credentials, so the file it names is read by
whoever names it. Ingestion must refuse cognee's own storage as a source."""

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognee.infrastructure.databases.cache.config as cache_config
import cognee.infrastructure.databases.graph.config as graph_config
import cognee.infrastructure.databases.relational.config as relational_config
import cognee.infrastructure.databases.vector.config as vector_config
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.files.utils import local_path_safety
from cognee.infrastructure.files.utils.local_path_safety import ALLOWED_LOCAL_FILE_ROOTS_ENV
from cognee.tasks.ingestion.create_dlt_source import (
    ALLOW_DATABASE_INGESTION_ENV,
    _allowed_sqlite_connection_string,
    create_dlt_source_from_connection_string,
)


@pytest.fixture
def storage(monkeypatch, tmp_path: Path) -> SimpleNamespace:
    """Point every cognee storage location at tmp_path, with the relational
    database outside the system root the way a custom DB_PATH puts it."""
    roots = SimpleNamespace(
        system=tmp_path / "system",
        db_dir=tmp_path / "custom_db_path",
        cache_db=tmp_path / "elsewhere" / "cache.db",
        outside=tmp_path / "outside",
    )
    for directory in (roots.system, roots.db_dir, roots.cache_db.parent, roots.outside):
        directory.mkdir(parents=True)

    base = SimpleNamespace(
        data_root_directory=str(tmp_path / "data"),
        system_root_directory=str(roots.system),
        cache_root_directory=str(tmp_path / "cache"),
        logs_root_directory=str(tmp_path / "logs"),
        repos_root_directory=str(tmp_path / "repos"),
    )
    monkeypatch.setattr(local_path_safety, "get_base_config", lambda: base)
    monkeypatch.setattr(
        relational_config,
        "get_relational_config",
        lambda: SimpleNamespace(db_path=str(roots.db_dir)),
    )
    monkeypatch.setattr(
        graph_config, "get_graph_config", lambda: SimpleNamespace(graph_file_path="")
    )
    monkeypatch.setattr(
        vector_config,
        "get_vectordb_config",
        lambda: SimpleNamespace(vector_db_url="postgresql://cognee@db/cognee"),
    )
    monkeypatch.setattr(
        cache_config,
        "get_cache_config",
        lambda: SimpleNamespace(cache_db_url=f"sqlite+aiosqlite:///{roots.cache_db}"),
    )
    monkeypatch.delenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, raising=False)
    return roots


def _sqlite_file(path: Path) -> Path:
    sqlite3.connect(path).close()
    return path


def test_ordinary_sqlite_file_is_allowed_and_made_absolute(storage, monkeypatch):
    _sqlite_file(storage.outside / "shop.db")
    monkeypatch.chdir(storage.outside)

    allowed = _allowed_sqlite_connection_string("sqlite:///shop.db")

    assert allowed == "sqlite:///" + os.path.realpath(storage.outside / "shop.db")


@pytest.mark.parametrize(
    "location",
    [
        lambda s: s.system / "databases" / "cognee_db",
        lambda s: s.db_dir / "cognee_db",  # DB_PATH outside the system root
        lambda s: s.cache_db,  # CACHE_DB_URL pointing elsewhere
    ],
    ids=["system-root", "custom-db-path", "cache-db-url"],
)
def test_cognee_storage_is_refused(storage, location):
    target = location(storage)
    target.parent.mkdir(parents=True, exist_ok=True)
    _sqlite_file(target)

    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string(f"sqlite:///{target}")


def test_relative_path_into_cognee_storage_is_refused(storage, monkeypatch):
    _sqlite_file(storage.db_dir / "cognee_db")
    monkeypatch.chdir(storage.outside)

    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string("sqlite:///../custom_db_path/cognee_db")


def test_symlink_to_cognee_storage_is_refused(storage):
    target = _sqlite_file(storage.db_dir / "cognee_db")
    link = storage.outside / "innocent.db"
    link.symlink_to(target)

    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string(f"sqlite:///{link}")


def test_uri_option_is_refused(storage):
    _sqlite_file(storage.outside / "shop.db")

    with pytest.raises(CogneeValidationError, match="'uri' option"):
        _allowed_sqlite_connection_string(f"sqlite:///{storage.outside / 'shop.db'}?uri=true")


def test_allowlist_applies_to_sqlite_files(storage, monkeypatch, tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    monkeypatch.setenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, str(allowed_root))
    _sqlite_file(storage.outside / "shop.db")

    with pytest.raises(CogneeValidationError, match="COGNEE_ALLOWED_LOCAL_FILE_ROOTS"):
        _allowed_sqlite_connection_string(f"sqlite:///{storage.outside / 'shop.db'}")


def test_allowlist_never_admits_cognee_storage(storage, monkeypatch):
    """The allowlist always admits cognee's own roots, so the refusal must run first."""
    monkeypatch.setenv(ALLOWED_LOCAL_FILE_ROOTS_ENV, str(storage.outside))
    target = storage.system / "cognee_db"
    _sqlite_file(target)

    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string(f"sqlite:///{target}")


def test_in_memory_database_is_allowed(storage):
    assert _allowed_sqlite_connection_string("sqlite:///") == "sqlite:///"


def test_a_missing_file_is_refused_and_not_created(storage, monkeypatch):
    """The driver creates a missing file when it connects, which would let a
    caller leave empty files wherever the server can write."""
    monkeypatch.delenv(ALLOW_DATABASE_INGESTION_ENV, raising=False)
    target = storage.outside / "new.db"

    with pytest.raises(CogneeValidationError, match="existing database file"):
        create_dlt_source_from_connection_string(f"sqlite:///{target}")

    assert not target.exists()


def test_a_directory_is_refused(storage):
    with pytest.raises(CogneeValidationError, match="existing database file"):
        _allowed_sqlite_connection_string(f"sqlite:///{storage.outside}")


def test_a_missing_file_in_cognee_storage_is_refused_as_storage(storage):
    """The storage refusal comes first, so a probe into cognee's storage learns
    nothing about which files exist there."""
    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string(f"sqlite:///{storage.system / 'nothing_here.db'}")


def test_a_vector_url_that_is_not_a_sqlalchemy_url_does_not_break_the_check(storage, monkeypatch):
    """Neptune Analytics is configured as `neptune-graph://<id>`, which make_url
    rejects; the internal-roots check must still run (and still refuse)."""
    monkeypatch.setattr(
        vector_config,
        "get_vectordb_config",
        lambda: SimpleNamespace(vector_db_url="neptune-graph://g-abc123"),
    )
    _sqlite_file(storage.outside / "shop.db")
    _allowed_sqlite_connection_string(f"sqlite:///{storage.outside / 'shop.db'}")
    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        _allowed_sqlite_connection_string(f"sqlite:///{storage.system / 'cognee_db'}")


@pytest.mark.parametrize("value", ["true", "1"])
def test_cognee_storage_stays_refused_with_the_switch_on(storage, monkeypatch, value):
    """The switch opens the feature; it never opens cognee's own storage."""
    monkeypatch.setenv(ALLOW_DATABASE_INGESTION_ENV, value)
    target = _sqlite_file(storage.db_dir / "cognee_db")

    with pytest.raises(CogneeValidationError, match="cognee's own storage"):
        create_dlt_source_from_connection_string(f"sqlite:///{target}")


def test_driver_errors_are_reported_by_class_only(storage, monkeypatch):
    """The advisory's probe: a file that is not a database. The driver's message
    (which carries the SQL it ran and says why the open failed) stays in the log;
    the caller learns the exception class and nothing else."""
    monkeypatch.delenv(ALLOW_DATABASE_INGESTION_ENV, raising=False)
    target = storage.outside / "hostname"
    target.write_text("not a database\n")

    with pytest.raises(CogneeValidationError) as info:
        create_dlt_source_from_connection_string(f"sqlite:///{target}")

    assert info.value.name == "DatabaseSourceUnreachable"
    assert "DatabaseError" in info.value.message
    assert "sqlite_master" not in info.value.message
    assert "not a database" not in info.value.message
