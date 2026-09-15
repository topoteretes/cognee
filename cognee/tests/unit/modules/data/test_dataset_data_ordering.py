"""Regression test: dataset listings are ordered by an indexed column.

get_dataset_data() used to sort by ``data_size DESC``. That column has no
index, and DataDTO does not even expose it -- so every listing seq-scanned
``data`` and sorted it externally to produce an order nobody could see. A LIMIT
does not avoid the sort: the whole partition has to be ordered to find the top
N. It is now ``created_at DESC, id``, which ``ix_data_dataset_created`` serves
end to end.
"""

import re

import pytest
from sqlalchemy import create_engine, create_mock_engine, inspect, text
from sqlalchemy.dialects import postgresql, sqlite

from cognee.modules.data.models import Data

INDEX_NAME = "ix_data_dataset_created"


def _listing_index():
    for index in Data.__table__.indexes:
        if index.name == INDEX_NAME:
            return index
    return None


def test_model_declares_the_listing_index():
    assert _listing_index() is not None, f"{INDEX_NAME} missing from Data.__table_args__"


@pytest.mark.parametrize("dialect", [postgresql.dialect(), sqlite.dialect()])
def test_index_covers_filter_and_sort_in_order(dialect):
    """Column order is what makes it usable: filter first, then the sort keys."""
    statements = []
    engine = create_mock_engine(
        f"{dialect.name}://",
        lambda ddl, *a, **kw: statements.append(str(ddl.compile(dialect=dialect))),
    )
    Data.__table__.create(engine)
    indexes = [ddl for ddl in statements if INDEX_NAME in ddl]
    assert len(indexes) == 1
    nulls = " NULLS LAST" if dialect.name == "postgresql" else ""
    assert f"(dataset_id, created_at DESC{nulls}, id)" in " ".join(indexes[0].split())


@pytest.mark.asyncio
async def test_method_issues_the_indexed_ordering(monkeypatch):
    """Reads the ORDER BY off the statement get_dataset_data actually builds.

    Compiled SQL, not the module source: a comment naming the old column would
    satisfy a text search while the query still sorted the wrong way.
    """
    import uuid
    from contextlib import asynccontextmanager

    import importlib

    # import_module, not `from ... import get_dataset_data`: the package
    # re-exports the function under the module's own name, so the plain import
    # binds the function and monkeypatching it would silently do nothing.
    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")

    captured = {}

    class _Session:
        async def execute(self, statement):
            captured["sql"] = " ".join(str(statement.compile(dialect=postgresql.dialect())).split())

            class _Result:
                def scalars(self):
                    class _S:
                        def all(self_inner):
                            return []

                    return _S()

            return _Result()

    class _Engine:
        @asynccontextmanager
        async def get_async_session(self):
            yield _Session()

    monkeypatch.setattr(module, "get_relational_engine", lambda: _Engine())

    await module.get_dataset_data(uuid.uuid4(), order_by="created_at")

    order_by = re.search(r"ORDER BY (.+?)(?: LIMIT|$)", captured["sql"]).group(1).strip()
    assert "data_size" not in order_by, "the unindexed column must be gone from the sort"
    assert "created_at DESC NULLS LAST" in order_by
    assert order_by.endswith("data.id"), "id tiebreak keeps paging stable"


def _migration():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "alembic/versions/e7f9a1c3d5b8_add_data_dataset_created_index.py"
    )
    spec = importlib.util.spec_from_file_location("listing_index_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sqlite_migration_upgrade_retry_and_downgrade():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE data (id TEXT, dataset_id TEXT, created_at DATETIME)"))
            with Operations.context(MigrationContext.configure(conn)):
                migration = _migration()
                migration.upgrade()
                migration.upgrade()
                ddl = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE name = :name"), {"name": INDEX_NAME}
                ).scalar_one()
                assert "(dataset_id, created_at DESC, id)" in ddl
                migration.downgrade()
                migration.downgrade()
                assert not inspect(conn).get_indexes("data")
    finally:
        engine.dispose()


@pytest.mark.parametrize("valid", [None, False, True])
def test_postgres_migration_repairs_index_without_releasing_transaction(monkeypatch, valid):
    from types import SimpleNamespace
    from unittest.mock import Mock

    migration = _migration()
    connection = Mock()
    connection.dialect.name = "postgresql"
    connection.execute.return_value.scalar_one_or_none.return_value = valid
    monkeypatch.setattr(
        migration, "inspect", lambda conn: SimpleNamespace(get_table_names=lambda: ["data"])
    )
    execute = Mock()
    # No autocommit API: using it would release the upgrader's version-row lock.
    monkeypatch.setattr(
        migration, "op", SimpleNamespace(get_bind=lambda: connection, execute=execute)
    )
    migration.upgrade()
    statements = [call.args[0] for call in execute.call_args_list]
    if valid is True:
        assert statements == []
    else:
        expected = [
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON data (dataset_id, created_at DESC NULLS LAST, id)"
        ]
        if valid is False:
            expected.insert(0, f"DROP INDEX IF EXISTS {INDEX_NAME}")
        assert statements == expected


def test_migration_skips_absent_data_table():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    engine = create_engine("sqlite://")
    try:
        with engine.begin() as conn, Operations.context(MigrationContext.configure(conn)):
            _migration().upgrade()
            _migration().downgrade()
            assert inspect(conn).get_table_names() == []
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_pages_preserve_timestamp_ties_and_dataset_isolation(monkeypatch):
    import importlib
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from uuid import UUID

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    module = importlib.import_module("cognee.modules.data.methods.get_dataset_data")
    engine = create_async_engine("sqlite+aiosqlite://")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        module, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    base = UUID("aaaaaaaa-0000-0000-0000-000000000000").int
    dataset_id = UUID(int=base + 100)
    now = datetime.now(timezone.utc)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Data.__table__.create)
        async with sessions() as session:
            session.add_all(
                [
                    Data(
                        id=UUID(int=base + i),
                        dataset_id=dataset_id,
                        created_at=created,
                        data_size=size,
                    )
                    for i, created, size in [
                        (3, now, 900),
                        (1, now, 1),
                        (4, now - timedelta(days=1), 5000),
                        (2, now, 20),
                    ]
                ]
            )
            session.add(
                Data(id=UUID(int=base + 5), dataset_id=UUID(int=base + 101), created_at=now)
            )
            await session.commit()
        # A legacy writer may omit the ORM's timestamp default.
        async with sessions() as session:
            from sqlalchemy import update

            await session.execute(
                update(Data).where(Data.id == UUID(int=base + 4)).values(created_at=None)
            )
            await session.commit()
        pages = [
            await module.get_dataset_data(dataset_id, limit=2, offset=o, order_by="created_at")
            for o in (0, 2, 4)
        ]
        assert [len(page) for page in pages] == [2, 2, 0]
        assert [row.id.int - base for page in pages for row in page] == [1, 2, 3, 4]
        assert [row.id.int - base for row in await module.get_dataset_data(dataset_id)] == [
            4,
            3,
            2,
            1,
        ]
    finally:
        await engine.dispose()
