"""Integration tests for chain-born relational databases (real SQLite).

The fresh-database bootstrap no longer runs ``create_all`` + ``stamp head`` —
``create_database()`` prepares the storage and builds the schema by running
the Alembic chain, whose initial revision carries the frozen base schema.
These tests pin the three properties that change makes load-bearing:

1. the chain alone builds the COMPLETE schema on an empty database — every
   model table and column, nothing stamped that did not execute;
2. the build is idempotent (every revision is existence-guarded);
3. a legacy database built by ``create_all`` with no alembic bookkeeping is
   healed by the same chain: replaying from base no-ops per guarded revision
   and installs honest ``alembic_version`` bookkeeping.
"""

import asyncio
import importlib
import tempfile
from pathlib import Path

import sqlalchemy as sa

# The one chain-built table with no model: the Notebook model was removed
# without its drop migration ever being written. Chain truth includes it.
CHAIN_ONLY_TABLES = {"alembic_version", "notebooks"}


def _register_all_models() -> None:
    """Import every ``cognee.modules.<x>.models`` package so ``Base.metadata``
    is the full certified surface, not whatever this process touched first."""
    import cognee.modules

    root = Path(cognee.modules.__file__).parent
    for sub in sorted(path.name for path in root.iterdir() if path.is_dir()):
        if (root / sub / "models" / "__init__.py").exists():
            importlib.import_module(f"cognee.modules.{sub}.models")


def _point_cognee_at_fresh_dirs() -> None:
    import cognee
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    root = tempfile.mkdtemp()
    cognee.config.data_root_directory(f"{root}/data")
    cognee.config.system_root_directory(f"{root}/system")
    create_relational_engine.cache_clear()


async def _reflect() -> dict[str, set[str]]:
    from cognee.infrastructure.databases.relational import get_relational_engine

    async with get_relational_engine().engine.connect() as connection:
        return await connection.run_sync(
            lambda sync_conn: {
                table: {column["name"] for column in sa.inspect(sync_conn).get_columns(table)}
                for table in sa.inspect(sync_conn).get_table_names()
            }
        )


def _model_schema() -> dict[str, set[str]]:
    _register_all_models()
    from cognee.infrastructure.databases.relational import Base

    return {
        table.name: {column.name for column in table.columns}
        for table in Base.metadata.sorted_tables
    }


def test_chain_builds_the_complete_schema_on_an_empty_database():
    from cognee.infrastructure.databases.relational import get_relational_engine

    expected = _model_schema()
    _point_cognee_at_fresh_dirs()
    asyncio.run(get_relational_engine().create_database())
    actual = asyncio.run(_reflect())

    missing_tables = set(expected) - set(actual)
    assert not missing_tables, f"chain never creates: {sorted(missing_tables)}"
    unexpected = set(actual) - set(expected) - CHAIN_ONLY_TABLES
    assert not unexpected, f"chain creates tables no model declares: {sorted(unexpected)}"
    column_diffs = {
        table: sorted(expected[table] ^ actual[table])
        for table in expected
        if expected[table] != actual[table]
    }
    assert not column_diffs, f"chain and models disagree on columns: {column_diffs}"
    assert "alembic_version" in actual, "the chain must install real bookkeeping"


def test_create_database_is_idempotent():
    from cognee.infrastructure.databases.relational import get_relational_engine

    _point_cognee_at_fresh_dirs()
    asyncio.run(get_relational_engine().create_database())
    before = asyncio.run(_reflect())
    asyncio.run(get_relational_engine().create_database())
    assert asyncio.run(_reflect()) == before


def test_create_database_builds_the_adapters_own_database():
    """A directly constructed adapter (the shape every unit-test fixture uses)
    must get its OWN database built — the chain runs against that adapter,
    not against whatever the global configuration points at."""
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    _point_cognee_at_fresh_dirs()  # the global engine points somewhere else
    own = Path(tempfile.mkdtemp()) / "own.db"
    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{own}")

    async def build_and_reflect():
        await adapter.create_database()
        async with adapter.engine.connect() as connection:
            return await connection.run_sync(lambda c: set(sa.inspect(c).get_table_names()))

    tables = asyncio.run(build_and_reflect())
    assert {"users", "data", "pipeline_runs", "alembic_version"} <= tables
    assert not asyncio.run(_reflect()), "the global database must be untouched"


def test_concurrent_creators_take_turns_on_the_alembic_command():
    """Every creation path now runs the chain, and callers CAN overlap
    (create_db_and_tables alongside run_migrations). Alembic's command
    machinery is process-global and not thread-safe — overlapping commands
    in worker threads corrupt each other's context — so concurrent creators
    must serialize, and each still ends at a complete, head-stamped schema."""
    from cognee.infrastructure.databases.relational import get_relational_engine

    _point_cognee_at_fresh_dirs()

    async def create_four_at_once():
        engine = get_relational_engine()
        await asyncio.gather(*(engine.create_database() for _ in range(4)))

    asyncio.run(create_four_at_once())
    built = asyncio.run(_reflect())
    assert "alembic_version" in built and "user_id" in built["pipeline_runs"]


def test_chain_heals_a_legacy_create_all_database_without_bookkeeping():
    """A database built by create_all with NO alembic_version (the legacy /
    pre-Alembic shape) replays the chain from base: every guarded revision
    no-ops against what exists, and the bookkeeping lands honestly."""
    from cognee.infrastructure.databases.relational import Base, get_relational_engine

    _register_all_models()
    _point_cognee_at_fresh_dirs()

    async def build_legacy_then_migrate():
        import os

        engine = get_relational_engine()
        # A legacy database exists on disk by definition; only create_database
        # preps storage, and this scenario deliberately bypasses it.
        os.makedirs(os.path.dirname(engine.db_path), exist_ok=True)
        async with engine.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        from cognee.modules.migrations.startup import run_relational_migrations

        await run_relational_migrations("head")

    asyncio.run(build_legacy_then_migrate())
    healed = asyncio.run(_reflect())
    assert "alembic_version" in healed
    assert "users" in healed and "user_id" in healed["pipeline_runs"]
