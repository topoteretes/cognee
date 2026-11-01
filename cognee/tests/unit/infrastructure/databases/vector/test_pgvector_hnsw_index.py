"""pgvector collections get an HNSW index, and search orders by the indexed expression.

Without an index every search is a full scan of every stored vector. pgvector indexes
``vector`` up to 2,000 dimensions and ``halfvec`` up to 4,000, so larger embeddings are
indexed through a halfvec cast, and the search must order by that same expression for
the planner to use the index. HNSW needs pgvector 0.5.0 and halfvec 0.7.0; on an older
server a collection keeps the exact scan.
"""

import pytest

pytest.importorskip("pgvector")

from types import SimpleNamespace

from pgvector.sqlalchemy import Vector
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


def test_small_embeddings_index_the_vector_column():
    assert hnsw_index_statement("Entity_name", 1536, CURRENT) == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector vector_cosine_ops)'
    )


def test_large_embeddings_index_a_halfvec_cast():
    assert hnsw_index_statement("Entity_name", 3072, CURRENT) == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw ((vector::halfvec(3072)) halfvec_cosine_ops)'
    )


def test_embeddings_pgvector_cannot_index_get_no_index():
    assert hnsw_index_statement("Entity_name", 4097, CURRENT) is None


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
    assert (hnsw_index_statement("t", dimensions, version) is None) == (expected is None)


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


def _compiled(dimensions):
    table = Table(
        "Entity_name",
        MetaData(),
        Column("id", String, primary_key=True),
        Column("vector", Vector(dimensions)),
    )
    indexed_type = _indexed_type(dimensions, CURRENT)
    expression = _indexed_distance(table.c.vector, [0.1] * dimensions, indexed_type, dimensions)
    return str(expression.compile(dialect=postgresql.dialect()))


def test_search_orders_by_the_indexed_expression():
    # The ORDER BY must match the index expression or Postgres cannot use the index.
    assert _compiled(3) == '"Entity_name".vector <=> %(vector_1)s'
    assert _compiled(3072) == 'CAST("Entity_name".vector AS HALFVEC(3072)) <=> %(param_1)s'


@pytest.mark.parametrize("version", [None, (0, 4, 0), (0, 6, 0), (0, 7, 0), CURRENT])
@pytest.mark.parametrize("dimensions", [1536, 3072, 4097])
@pytest.mark.parametrize("table_name", ["Entity_name", "B" * 70])
def test_migration_statements_match_the_adapter(version, dimensions, table_name):
    # The migration carries a frozen copy; it must build the index the adapter builds.
    assert pgvector_hnsw_indexes._frozen_index_statement(
        table_name, dimensions, version
    ) == hnsw_index_statement(table_name, dimensions, version)
    assert pgvector_hnsw_indexes._frozen_index_name(table_name) == hnsw_index_name(table_name)


@pytest.mark.asyncio
async def test_migration_is_a_noop_for_other_vector_backends():
    # No engine attribute is touched for a non-pgvector store.
    await pgvector_hnsw_indexes.migrate(SimpleNamespace(vector_engine=object()))
    await pgvector_hnsw_indexes.downgrade(SimpleNamespace(vector_engine=object()))
