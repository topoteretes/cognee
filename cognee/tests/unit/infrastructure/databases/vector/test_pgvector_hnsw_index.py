"""pgvector collections get an HNSW index, and search orders by the indexed expression.

Without an index every search is a full scan of every stored vector. pgvector indexes
``vector`` up to 2,000 dimensions and ``halfvec`` up to 4,000, so larger embeddings are
stored and indexed as ``halfvec``. A wide collection created with a full-precision
``vector`` column is indexed through a halfvec cast, and the search must order by that
same expression for the planner to use the index. HNSW needs pgvector 0.5.0 and halfvec
0.7.0; on an older server a collection keeps the exact scan.
"""

import pytest

pytest.importorskip("pgvector")

from contextlib import asynccontextmanager
from types import SimpleNamespace

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects import postgresql

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    _HNSW_EF_CONSTRUCTION,
    _HNSW_M,
    _column_type,
    _indexed_distance,
    _indexed_type,
    hnsw_index_name,
    hnsw_index_statement,
    parse_pgvector_version,
)
from cognee.modules.migrations.versions import pgvector_hnsw_indexes

CURRENT = (0, 8, 0)
# TEST ONLY: the build parameters new indexes get.
WITH = f" WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION})"


def test_small_embeddings_index_the_vector_column():
    assert _column_type(1536, CURRENT) == "vector"
    assert hnsw_index_statement("Entity_name", 1536, CURRENT, column_type="vector") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector vector_cosine_ops)' + WITH
    )


def test_large_embeddings_are_stored_as_halfvec_and_index_the_column():
    assert _column_type(3072, CURRENT) == "halfvec"
    assert hnsw_index_statement("Entity_name", 3072, CURRENT, column_type="halfvec") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector halfvec_cosine_ops)' + WITH
    )


def test_a_large_vector_column_indexes_a_halfvec_cast():
    # A wide collection created with a full-precision column keeps it.
    assert hnsw_index_statement("Entity_name", 3072, CURRENT, column_type="vector") == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw ((vector::halfvec(3072)) halfvec_cosine_ops)' + WITH
    )


def test_embeddings_pgvector_cannot_index_get_no_index():
    assert _column_type(4097, CURRENT) == "vector"
    assert hnsw_index_statement("Entity_name", 4097, CURRENT, column_type="vector") is None


@pytest.mark.parametrize(
    ("version", "dimensions", "expected"),
    [
        (None, 1536, None),  # extension not installed / unknown
        ((0, 4, 4), 1536, None),  # no HNSW before 0.5.0
        ((0, 5, 0), 1536, "vector"),
        ((0, 6, 2), 3072, None),  # no halfvec before 0.7.0: keep the exact scan
        ((0, 7, 0), 3072, "halfvec"),
        ((0, 8, 0), 0, None),
    ],
)
def test_the_index_follows_the_installed_pgvector(version, dimensions, expected):
    assert _indexed_type(dimensions, version) == expected
    column_type = _column_type(dimensions, version)
    assert column_type == ("halfvec" if expected == "halfvec" else "vector")
    statement = hnsw_index_statement("t", dimensions, version, column_type=column_type)
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
    indexed_type = _indexed_type(dimensions, CURRENT)
    expression = _indexed_distance(table.c.vector, [0.1] * dimensions, indexed_type, dimensions)
    return str(expression.compile(dialect=postgresql.dialect()))


def test_search_orders_by_the_indexed_expression():
    # The ORDER BY must match the index expression or Postgres cannot use the index.
    assert _compiled(3) == '"Entity_name".vector <=> %(vector_1)s'
    assert _compiled(3072) == 'CAST("Entity_name".vector AS HALFVEC(3072)) <=> %(param_1)s'
    # A halfvec column is indexed as it is: no cast.
    assert _compiled(3072, HALFVEC) == '"Entity_name".vector <=> %(vector_1)s'


@pytest.mark.parametrize("version", [None, (0, 4, 0), (0, 6, 0), (0, 7, 0), CURRENT])
@pytest.mark.parametrize("dimensions", [1536, 3072, 4097])
@pytest.mark.parametrize("table_name", ["Entity_name", "B" * 70])
def test_migration_statements_match_the_adapter(version, dimensions, table_name):
    # The migration carries a frozen copy; it must build the index the adapter builds for
    # the collections it sees, which are ``vector`` columns only (its catalog query).
    # TEST ONLY: the adapter now adds build parameters (``WITH``) the
    # frozen copy does not have; the index is otherwise the same (name, expression,
    # operator class), and both build only where none exists (``IF NOT EXISTS``).
    frozen = pgvector_hnsw_indexes._frozen_index_statement(table_name, dimensions, version)
    adapter = hnsw_index_statement(table_name, dimensions, version, column_type="vector")
    assert (frozen + WITH if frozen is not None else None) == adapter
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


def _assert_serial_build_in_raised_memory(statements, index_statement):
    # Docker caps /dev/shm at 64 MB; a parallel build at the raised maintenance_work_mem
    # needs a shared segment of that size and fails, so the build must be serial, and the
    # settings must come before the CREATE INDEX in its own transaction (SET LOCAL).
    assert statements == [
        (
            "SELECT set_config('maintenance_work_mem', :target, true) FROM pg_settings "
            "WHERE name = 'maintenance_work_mem' AND setting::bigint < :target_kb"
        ),
        "SET LOCAL max_parallel_maintenance_workers = 0",
        index_statement,
    ]


@pytest.mark.asyncio
async def test_adapter_rebuild_runs_a_serial_build_with_raised_memory():
    engine = _Engine()
    result = await _adapter_with(engine).run_migrations()
    assert result == {"hnsw_index_checked": ["Entity_name"], "hnsw_index_failed": []}
    _assert_serial_build_in_raised_memory(
        _build_transaction(engine.log),
        hnsw_index_statement("Entity_name", 3072, CURRENT, column_type="vector"),
    )


@pytest.mark.asyncio
async def test_migration_runs_a_serial_build_with_raised_memory():
    engine = _Engine()
    await pgvector_hnsw_indexes.migrate(SimpleNamespace(vector_engine=_adapter_with(engine)))
    _assert_serial_build_in_raised_memory(
        _build_transaction(engine.log),
        pgvector_hnsw_indexes._frozen_index_statement("Entity_name", 3072, CURRENT),
    )
