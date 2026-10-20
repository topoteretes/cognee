"""pgvector collections get an HNSW index, and search orders by the indexed expression.

Without an index every search is a full scan of every stored vector. pgvector indexes
``vector`` up to 2,000 dimensions and ``halfvec`` up to 4,000, so larger embeddings are
indexed through a halfvec cast, and the search must order by that same expression for
the planner to use the index.
"""

import pytest

pytest.importorskip("pgvector")

from types import SimpleNamespace

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, MetaData, String, Table
from sqlalchemy.dialects import postgresql

from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    _indexed_distance,
    hnsw_index_statement,
)
from cognee.modules.migrations.versions import pgvector_hnsw_indexes


def test_small_embeddings_index_the_vector_column():
    assert hnsw_index_statement("Entity_name", 1536) == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw (vector vector_cosine_ops)'
    )


def test_large_embeddings_index_a_halfvec_cast():
    assert hnsw_index_statement("Entity_name", 3072) == (
        'CREATE INDEX IF NOT EXISTS "Entity_name_vector_hnsw" '
        'ON "Entity_name" USING hnsw ((vector::halfvec(3072)) halfvec_cosine_ops)'
    )


def test_embeddings_pgvector_cannot_index_get_no_index():
    assert hnsw_index_statement("Entity_name", 4097) is None


def _compiled(dimensions):
    table = Table(
        "Entity_name",
        MetaData(),
        Column("id", String, primary_key=True),
        Column("vector", Vector(dimensions)),
    )
    expression = _indexed_distance(table.c.vector, [0.1] * dimensions, dimensions)
    return str(expression.compile(dialect=postgresql.dialect()))


def test_search_orders_by_the_indexed_expression():
    # The ORDER BY must match the index expression or Postgres cannot use the index.
    assert _compiled(3) == '"Entity_name".vector <=> %(vector_1)s'
    assert _compiled(3072) == 'CAST("Entity_name".vector AS HALFVEC(3072)) <=> %(param_1)s'


def test_migration_statements_match_the_adapter():
    # The migration carries a frozen copy; it must build the index the adapter builds.
    for dimensions in (1536, 3072, 4097):
        assert pgvector_hnsw_indexes._frozen_index_statement(
            "Entity_name", dimensions
        ) == hnsw_index_statement("Entity_name", dimensions)


@pytest.mark.asyncio
async def test_migration_is_a_noop_for_other_vector_backends():
    # No engine attribute is touched for a non-pgvector store.
    await pgvector_hnsw_indexes.migrate(SimpleNamespace(vector_engine=object()))
    await pgvector_hnsw_indexes.downgrade(SimpleNamespace(vector_engine=object()))
