"""Migration ``c3d5e7f9a1b2`` heals a standalone cache.db that lives on the Turso engine.

The migration dedupes ``cache_session_context`` and adds its unique index directly on
the cache file. For a Turso cache in ``mvcc`` mode that file is not readable by stdlib
``sqlite3`` at all, so the migration has to open it through pyturso.
"""

import importlib.util
import pathlib
import sqlite3

import pytest

pytest.importorskip("turso", reason="pyturso not installed")

import turso

# parents[5] is the ``cognee`` package directory, which holds ``alembic/``.
MIGRATION = (
    pathlib.Path(__file__).resolve().parents[5]
    / "alembic"
    / "versions"
    / "c3d5e7f9a1b2_dedupe_session_context_add_unique_index.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("cache_migration_c3d5e7f9a1b2", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_duplicates(connection, table: str) -> None:
    connection.execute(
        f"CREATE TABLE {table} (seq INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, "
        "session_id TEXT NOT NULL, entry_id TEXT NOT NULL, payload TEXT NOT NULL, "
        "created_at TEXT, expires_at TEXT)"
    )
    for payload in ("old", "newer", "newest"):
        connection.execute(
            f"INSERT INTO {table} (user_id, session_id, entry_id, payload) "
            f"VALUES ('u', 's', 'e1', '{payload}')"
        )
    connection.execute(
        f"INSERT INTO {table} (user_id, session_id, entry_id, payload) VALUES ('u', 's', 'e2', 'x')"
    )
    connection.commit()


@pytest.mark.parametrize("journal_mode", ["wal", "mvcc"])
def test_heal_turso_cache_in_both_journal_modes(tmp_path, journal_mode):
    migration = _load_migration()
    path = str(tmp_path / "cache.db")

    connection = turso.connect(path)
    connection.execute(f"PRAGMA journal_mode={journal_mode}").fetchall()
    _seed_duplicates(connection, migration.TABLE_NAME)
    connection.close()

    if journal_mode == "mvcc":
        # The reason the migration must not use stdlib sqlite3 for a Turso cache.
        with pytest.raises(sqlite3.DatabaseError):
            sqlite3.connect(path).execute("SELECT count(*) FROM sqlite_master").fetchone()

    migration.heal_standalone_sqlite_cache(path, migration.TURSO_DRIVER)

    connection = turso.connect(path)
    rows = connection.execute(
        f"SELECT entry_id, payload FROM {migration.TABLE_NAME} ORDER BY entry_id"
    ).fetchall()
    indexes = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    ]
    assert rows == [("e1", "newest"), ("e2", "x")]
    assert migration.INDEX_NAME in indexes

    # Idempotent: a second run is a no-op, and the downgrade mirror removes the index.
    migration.heal_standalone_sqlite_cache(path, migration.TURSO_DRIVER)
    migration.drop_standalone_sqlite_cache_index(path, migration.TURSO_DRIVER)
    indexes = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    ]
    assert migration.INDEX_NAME not in indexes
    connection.close()


def test_heal_sqlite_cache_still_uses_stdlib_driver(tmp_path):
    """The aiosqlite path is unchanged: a plain SQLite cache is healed through sqlite3."""
    migration = _load_migration()
    path = str(tmp_path / "cache.db")
    connection = sqlite3.connect(path)
    _seed_duplicates(connection, migration.TABLE_NAME)
    connection.close()

    migration.heal_standalone_sqlite_cache(path)  # default driver: aiosqlite

    connection = sqlite3.connect(path)
    assert connection.execute(f"SELECT count(*) FROM {migration.TABLE_NAME}").fetchone() == (2,)
    connection.close()


def test_standalone_cache_path_reports_the_driver(tmp_path, monkeypatch):
    migration = _load_migration()
    from cognee.infrastructure.databases.cache import config as cache_config_module
    from cognee.infrastructure.databases.turso import turso_url

    cache_file = tmp_path / "cache.db"
    cache_file.write_bytes(b"")
    fake_config = cache_config_module.CacheConfig(
        _env_file=None, cache_backend="turso", cache_db_url=turso_url(str(cache_file))
    )
    monkeypatch.setattr(cache_config_module, "get_cache_config", lambda: fake_config)

    class _FakeAlembicConn:
        class engine:
            from sqlalchemy.engine import make_url

            url = make_url(f"sqlite+aiosqlite:///{tmp_path}/cognee_db")

    assert migration.standalone_sqlite_cache_path(_FakeAlembicConn()) == (
        str(cache_file),
        migration.TURSO_DRIVER,
    )


def test_heal_turso_cache_without_driver_fails_with_install_hint(tmp_path):
    """A Turso cache with pyturso missing must fail loudly with the install hint."""
    import sys
    from unittest.mock import patch

    migration = _load_migration()
    path = str(tmp_path / "cache.db")
    with (
        patch.dict(sys.modules, {"turso": None}),
        pytest.raises(RuntimeError, match="pyturso is not installed"),
    ):
        migration.heal_standalone_sqlite_cache(path, migration.TURSO_DRIVER)
