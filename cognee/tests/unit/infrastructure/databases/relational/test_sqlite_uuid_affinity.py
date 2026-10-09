"""UUID columns are text on SQLite, on every path that creates a table.

SQLite gives a column declared ``UUID`` NUMERIC affinity, so a hex UUID made
only of digits is stored as a number and no longer equals its text form. Two
things keep that from happening: the models declare SQLAlchemy's portable
``Uuid`` (``CHAR(32)`` on SQLite), and ``sqlite_uuid`` makes the exact-name
``UUID`` the frozen base schema and the Alembic chain still declare compile
the same way. Postgres keeps its native type on both.
"""

from uuid import UUID

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from cognee.alembic.frozen_schema import frozen_metadata
from cognee.infrastructure.databases.relational import Base
from cognee.modules.migrations.lockstep import register_server_models

NUMERIC_LOOKING_UUID = UUID("12345678901234567890123456789012")


def test_exact_name_uuid_compiles_to_text_on_sqlite_only():
    assert sa.UUID().compile(dialect=sqlite.dialect()) == "CHAR(32)"
    assert sa.UUID().compile(dialect=postgresql.dialect()) == "UUID"


def test_every_frozen_schema_uuid_column_is_text_on_sqlite():
    """The initial revision builds fresh databases from this surface."""
    uuid_columns = [
        column
        for table in frozen_metadata("sqlite").tables.values()
        for column in table.columns
        if isinstance(column.type, sa.Uuid)
    ]
    assert uuid_columns
    assert {column.type.compile(dialect=sqlite.dialect()) for column in uuid_columns} == {
        "CHAR(32)"
    }


def test_numeric_looking_uuid_round_trips_through_a_frozen_table():
    metadata = frozen_metadata("sqlite")
    data = metadata.tables["data"]
    engine = sa.create_engine("sqlite://")
    data.create(engine)

    with engine.begin() as connection:
        connection.execute(sa.insert(data).values(id=NUMERIC_LOOKING_UUID))
        storage_type = connection.scalar(sa.text("SELECT typeof(id) FROM data"))
        stored = connection.scalar(sa.select(data.c.id).where(data.c.id == NUMERIC_LOOKING_UUID))

    assert storage_type == "text"
    assert stored == NUMERIC_LOOKING_UUID


@pytest.fixture(scope="module")
def all_models_registered():
    register_server_models()
    return Base.metadata


def test_models_declare_the_portable_uuid_type(all_models_registered):
    """New models must use ``Uuid``; the exact-name ``UUID`` only survives in the chain."""
    exact_name = sorted(
        f"{table.name}.{column.name}"
        for table in all_models_registered.tables.values()
        for column in table.columns
        if isinstance(column.type, sa.Uuid) and type(column.type) is not sa.Uuid
    )
    assert exact_name == []


def test_models_store_uuids_as_text_on_sqlite(all_models_registered):
    uuid_columns = [
        column
        for table in all_models_registered.tables.values()
        for column in table.columns
        if isinstance(column.type, sa.Uuid)
    ]
    assert len(uuid_columns) > 50
    assert {column.type.compile(dialect=sqlite.dialect()) for column in uuid_columns} == {
        "CHAR(32)"
    }
    assert {column.type.compile(dialect=postgresql.dialect()) for column in uuid_columns} == {
        "UUID"
    }
