"""Exercise the real Alembic chain on a local URL containing a literal percent."""

import asyncio

from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.migrations.startup import _build_alembic_config


def test_create_database_migrates_percent_path_and_can_run_again(tmp_path):
    database = tmp_path / "user%40example.db"
    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{database.as_posix()}")

    async def migrate():
        try:
            await adapter.create_database()
            await adapter.create_database()
            async with adapter.engine.connect() as connection:
                tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
                revision = (
                    await connection.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one()
            assert "users" in tables
            assert (
                revision == ScriptDirectory.from_config(_build_alembic_config()).get_current_head()
            )
        finally:
            await adapter.engine.dispose()

    asyncio.run(migrate())
    assert database.is_file()
