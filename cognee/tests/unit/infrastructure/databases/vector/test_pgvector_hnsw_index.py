"""pgvector collections get an HNSW index, and search orders by the indexed expression.

Without an index every search is a full scan of every stored vector. A new collection
stores ``halfvec``, indexed on the column up to 4,000 dimensions (pgvector's HNSW limit for
halfvec). A collection created by an earlier release with a ``vector`` column keeps it:
pgvector indexes ``vector`` up to 2,000 dimensions, so a wider one is indexed through a
halfvec cast, and the search must order by that same expression for the planner to use the
index. The adapter requires pgvector 0.7.0, the first release with halfvec.
"""

import pytest

pytest.importorskip("pgvector")

from contextlib import asynccontextmanager
from types import SimpleNamespace

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects import postgresql

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    _indexed_distance,
    _indexed_type,
    hnsw_index_name,
    hnsw_index_statement,
    parse_pgvector_version,
)
from cognee.modules.migrations.versions import pgvector_hnsw_indexes

CURRENT = (0, 8, 0)


@pytest.mark.parametrize("dimensions", [8, 1536, 3072, 4000])
def test_a_halfvec_column_is_indexed_on_the_column(dimensions):
    assert hnsw_index_statement("Entity_name", dimensions, column_type="halfvec") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector halfvec_cosine_ops)'
    )


def test_a_halfvec_column_pgvector_cannot_index_gets_no_index():
    assert hnsw_index_statement("Entity_name", 4001, column_type="halfvec") is None


def test_a_small_vector_column_is_indexed_on_the_column():
    assert hnsw_index_statement("Entity_name", 1536, column_type="vector") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector vector_cosine_ops)'
    )


def test_a_large_vector_column_is_indexed_through_a_halfvec_cast():
    assert hnsw_index_statement("Entity_name", 3072, column_type="vector") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw ((vector::halfvec(3072)) halfvec_cosine_ops)'
    )


def test_a_vector_column_pgvector_cannot_index_gets_no_index():
    assert hnsw_index_statement("Entity_name", 4097, column_type="vector") is None


@pytest.mark.parametrize(
    ("column_type", "dimensions", "expected"),
    [
        ("vector", 1536, "vector"),
        ("vector", 2000, "vector"),
        ("vector", 2001, "halfvec"),  # through a halfvec cast
        ("vector", 4001, None),
        ("vector", 0, None),  # undeclared width
        ("halfvec", 8, "halfvec"),
        ("halfvec", 1536, "halfvec"),
        ("halfvec", 4000, "halfvec"),
        ("halfvec", 4001, None),
        ("halfvec", 0, None),
    ],
)
def test_the_index_follows_the_stored_type(column_type, dimensions, expected):
    assert _indexed_type(dimensions, column_type) == expected
    statement = hnsw_index_statement("t", dimensions, column_type=column_type)
    assert (statement is None) == (expected is None)


def test_pgvector_version_parsing():
    assert parse_pgvector_version("0.8.0") == (0, 8, 0)
    assert parse_pgvector_version("0.7.4") == (0, 7, 4)
    assert parse_pgvector_version("0.5") == (0, 5)
    assert parse_pgvector_version(None) is None
    assert parse_pgvector_version("") is None


def test_long_collection_names_get_distinct_index_names():
    # Postgres truncates identifiers to 63 bytes; two long names sharing a prefix must
    # not collapse to one index name, or IF NOT EXISTS skips the second table.
    first = "A" * 60 + "_one"
    second = "A" * 60 + "_two"
    assert hnsw_index_name(first) != hnsw_index_name(second)
    for name in (first, second, "é" * 40):
        assert len(hnsw_index_name(name).encode()) <= 63
    assert hnsw_index_name("Entity_name") == "Entity_name_vector_hnsw"


def _compiled(dimensions, column_type=Vector):
    table = Table(
        "Entity_name",
        MetaData(),
        Column("id", String, primary_key=True),
        Column("vector", column_type(dimensions)),
    )
    stored = "halfvec" if column_type is HALFVEC else "vector"
    indexed_type = _indexed_type(dimensions, stored)
    expression = _indexed_distance(table.c.vector, [0.1] * dimensions, indexed_type, dimensions)
    return str(expression.compile(dialect=postgresql.dialect()))


def test_search_orders_by_the_indexed_expression():
    # The ORDER BY must match the index expression or Postgres cannot use the index.
    assert _compiled(3) == '"Entity_name".vector <=> %(vector_1)s'
    assert _compiled(3072) == 'CAST("Entity_name".vector AS HALFVEC(3072)) <=> %(param_1)s'
    # A halfvec column is indexed as it is, at every size: no cast.
    assert _compiled(3, HALFVEC) == '"Entity_name".vector <=> %(vector_1)s'
    assert _compiled(3072, HALFVEC) == '"Entity_name".vector <=> %(vector_1)s'


# The adapter runs on pgvector 0.7.0 or newer only (older raises PgvectorVersionError),
# so the frozen statement is compared where both run.
@pytest.mark.parametrize("version", [(0, 7, 0), CURRENT])
@pytest.mark.parametrize("dimensions", [1536, 3072, 4097])
@pytest.mark.parametrize("table_name", ["Entity_name", "B" * 70])
def test_migration_statements_match_the_adapter(version, dimensions, table_name):
    # The migration carries a frozen copy; it must build the index the adapter builds for
    # the collections it sees, which are ``vector`` columns only (its catalog query reads
    # no other type, so a ``halfvec`` collection is left to the adapter).
    assert "t.typname = 'vector'" in pgvector_hnsw_indexes._COLLECTIONS_SQL
    assert pgvector_hnsw_indexes._frozen_index_statement(
        table_name, dimensions, version
    ) == hnsw_index_statement(table_name, dimensions, column_type="vector")
    assert pgvector_hnsw_indexes._frozen_index_name(table_name) == hnsw_index_name(table_name)


@pytest.mark.asyncio
async def test_migration_is_a_noop_for_other_vector_backends():
    # No engine attribute is touched for a non-pgvector store.
    await pgvector_hnsw_indexes.migrate(SimpleNamespace(vector_engine=object()))
    await pgvector_hnsw_indexes.downgrade(SimpleNamespace(vector_engine=object()))


class _Result:
    def __init__(self, rows=(), scalar=None):
        self._rows = list(rows)
        self._scalar = scalar

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar

    def scalar(self):
        return self._scalar


class _Connection:
    """Records every statement; answers the catalog queries with one 3072-dim ``vector``
    table: (name, dimensions) for the frozen migration's query, (name, type, dimensions)
    for the adapter's."""

    def __init__(self, log):
        self.log = log

    async def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.log.append(sql)
        if "pg_attribute" in sql and "t.typname IN" in sql:
            return _Result(rows=[("Entity_name", "vector", 3072)])
        if "pg_attribute" in sql:
            return _Result(rows=[("Entity_name", 3072)])
        if "pg_extension" in sql:
            return _Result(scalar="0.8.0")
        return _Result()


class _Engine:
    def __init__(self):
        self.log = []

    @asynccontextmanager
    async def connect(self):
        yield _Connection(self.log)

    @asynccontextmanager
    async def begin(self):
        self.log.append("BEGIN")
        yield _Connection(self.log)
        self.log.append("COMMIT")


def _adapter_with(engine):
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    adapter.engine = engine
    adapter._pgvector_version_cache = None
    return adapter


def _build_transaction(log):
    start = log.index("BEGIN")
    return log[start + 1 : log.index("COMMIT")]


def _assert_serial_build_in_raised_memory(statements):
    # Docker caps /dev/shm at 64 MB; a parallel build at the raised maintenance_work_mem
    # needs a shared segment of that size and fails, so the build must be serial, and the
    # settings must come before the CREATE INDEX in its own transaction (SET LOCAL).
    assert statements == [
        (
            "SELECT set_config('maintenance_work_mem', :target, true) FROM pg_settings "
            "WHERE name = 'maintenance_work_mem' AND setting::bigint < :target_kb"
        ),
        "SET LOCAL max_parallel_maintenance_workers = 0",
        hnsw_index_statement("Entity_name", 3072, column_type="vector"),
    ]


@pytest.mark.asyncio
async def test_adapter_rebuild_runs_a_serial_build_with_raised_memory():
    engine = _Engine()
    result = await _adapter_with(engine).run_migrations()
    assert result == {"hnsw_index_checked": ["Entity_name"], "hnsw_index_failed": []}
    _assert_serial_build_in_raised_memory(_build_transaction(engine.log))


@pytest.mark.asyncio
async def test_migration_runs_a_serial_build_with_raised_memory():
    engine = _Engine()
    await pgvector_hnsw_indexes.migrate(SimpleNamespace(vector_engine=_adapter_with(engine)))
    _assert_serial_build_in_raised_memory(_build_transaction(engine.log))


class _VersionConnection:
    """Answers the pgvector version query with ``version``; anything else returns nothing."""

    def __init__(self, version):
        self.version = version
        self.log = []

    async def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.log.append(sql)
        if "pg_extension" in sql:
            return _Result(scalar=self.version)
        return _Result()


@pytest.mark.parametrize("version", ["0.6.2", "0.5.1", "0.4.4"])
@pytest.mark.asyncio
async def test_an_outdated_pgvector_raises_a_clear_error(version):
    from cognee.infrastructure.databases.vector.exceptions import PgvectorVersionError

    adapter = _adapter_with(_Engine())
    with pytest.raises(PgvectorVersionError) as raised:
        await adapter._pgvector_version(_VersionConnection(version))
    message = str(raised.value)
    assert f"version {version}" in message
    assert "0.7.0 or newer" in message
    assert "ALTER EXTENSION vector UPDATE" in message
    # Not cached: an upgraded extension is seen without a restart.
    assert adapter._pgvector_version_cache is None
    assert await adapter._pgvector_version(_VersionConnection("0.7.0")) == (0, 7, 0)


@pytest.mark.asyncio
async def test_an_unknown_pgvector_version_is_not_an_error():
    # Before CREATE EXTENSION the version is unknown; create_collection installs it first.
    adapter = _adapter_with(_Engine())
    assert await adapter._pgvector_version(_VersionConnection(None)) is None
    assert adapter._pgvector_version_cache is None


class _OutdatedConnection(_VersionConnection):
    """pgvector 0.6.2 with one 1536-dimensional ``vector`` collection in the catalog."""

    def __init__(self, log):
        super().__init__("0.6.2")
        self.log = log

    async def execute(self, statement, params=None):
        if "pg_attribute" in str(statement):
            self.log.append("catalog")
            return _Result(rows=[("Entity_name", "vector", 1536)])
        return await super().execute(statement, params)

    async def run_sync(self, fn, *args, **kwargs):
        self.log.append("run_sync")


class _OutdatedEngine(_Engine):
    @asynccontextmanager
    async def connect(self):
        yield _OutdatedConnection(self.log)

    @asynccontextmanager
    async def begin(self):
        yield _OutdatedConnection(self.log)


def _assert_outdated(raised):
    assert "version 0.6.2" in str(raised.value)
    assert "0.7.0 or newer" in str(raised.value)


@pytest.mark.asyncio
async def test_an_outdated_pgvector_fails_the_index_check():
    from cognee.infrastructure.databases.vector.exceptions import PgvectorVersionError

    engine = _OutdatedEngine()
    with pytest.raises(PgvectorVersionError) as raised:
        await _adapter_with(engine).run_migrations()
    _assert_outdated(raised)
    assert not any("CREATE INDEX" in sql for sql in engine.log)


@pytest.mark.asyncio
async def test_an_outdated_pgvector_fails_creating_a_collection_before_any_table():
    from unittest.mock import AsyncMock

    from cognee.infrastructure.databases.relational.ModelBase import Base
    from cognee.infrastructure.databases.vector.exceptions import PgvectorVersionError
    from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticLock

    engine = _OutdatedEngine()
    adapter = _adapter_with(engine)
    adapter.has_collection = AsyncMock(return_value=False)
    adapter.VECTOR_DB_LOCK = LoopAgnosticLock()
    adapter.embedding_engine = SimpleNamespace(get_vector_size=lambda: 8)
    collection = "OutdatedPgvector_text"
    try:
        with pytest.raises(PgvectorVersionError) as raised:
            await adapter.create_collection(collection)
    finally:
        # create_collection declares the table on the shared declarative Base.
        table = Base.metadata.tables.get(collection)
        if table is not None:
            Base.metadata.remove(table)
    _assert_outdated(raised)
    # The extension statement ran first (it must, on a database without pgvector), and
    # neither the table nor its index was created.
    assert engine.log[0] == "CREATE EXTENSION IF NOT EXISTS vector;"
    assert "run_sync" not in engine.log
    assert not any("CREATE INDEX" in sql for sql in engine.log)


@pytest.mark.asyncio
async def test_an_outdated_pgvector_fails_a_search():
    from unittest.mock import AsyncMock

    from cognee.infrastructure.databases.vector.exceptions import PgvectorVersionError

    engine = _OutdatedEngine()
    adapter = _adapter_with(engine)
    table = Table(
        "Entity_name",
        MetaData(),
        Column("id", String, primary_key=True),
        Column("vector", Vector(3)),
    )
    adapter.get_table = AsyncMock(return_value=table)
    adapter.embedding_engine = SimpleNamespace(get_vector_size=lambda: 3)

    @asynccontextmanager
    async def session():
        yield _OutdatedConnection(engine.log)

    adapter.get_async_session = session
    with pytest.raises(PgvectorVersionError) as raised:
        await adapter.search("Entity_name", query_vector=[0.1, 0.2, 0.3], limit=5)
    _assert_outdated(raised)
    assert not any("ORDER BY" in sql for sql in engine.log)
