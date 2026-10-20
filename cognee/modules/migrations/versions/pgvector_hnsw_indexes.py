"""Add an HNSW index to every pgvector collection that was created without one.

New collections get the index from ``PGVectorAdapter.create_collection``. Collections
created by earlier releases have only a primary key, so every search over them is a full
scan that reads every stored vector: a 3072-dimensional collection of 12k rows is ~830 MB
of pages per query, 20 s when the pages are not cached. This builds the same index for
them.

Only the pgvector backend is affected; for every other vector backend this is a no-op.
The DDL is a frozen snapshot of ``hnsw_index_statement`` (a migration is immutable history,
so it does not import the adapter's helper). Statements are idempotent (``IF NOT EXISTS``)
and the store is read from the Postgres catalog, so an empty store does nothing. Tables are
found in ``current_schema()``, which is the dataset's own schema in shared-database mode
(the adapter pins ``search_path``) and ``public`` otherwise.
"""

from __future__ import annotations

from sqlalchemy import text

from cognee.modules.migrations.migration import MigrationContext

_FROZEN_VECTOR_MAX_DIMENSIONS = 2000
_FROZEN_HALFVEC_MAX_DIMENSIONS = 4000

# Collection tables: a ``vector`` column of pgvector's ``vector`` type. For that type
# ``atttypmod`` is the declared dimension count (-1 when undeclared).
_COLLECTIONS_SQL = """
SELECT c.relname, a.atttypmod
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
WHERE n.nspname = current_schema()
  AND c.relkind = 'r'
  AND a.attname = 'vector'
  AND t.typname = 'vector'
  AND NOT a.attisdropped
"""


def _frozen_index_statement(table_name: str, dimensions: int) -> str | None:
    if dimensions <= 0:
        return None
    if dimensions <= _FROZEN_VECTOR_MAX_DIMENSIONS:
        indexed, operator_class = "vector", "vector_cosine_ops"
    elif dimensions <= _FROZEN_HALFVEC_MAX_DIMENSIONS:
        indexed, operator_class = f"(vector::halfvec({dimensions}))", "halfvec_cosine_ops"
    else:
        return None
    return (
        f'CREATE INDEX IF NOT EXISTS "{table_name}_vector_hnsw" '
        f'ON "{table_name}" USING hnsw ({indexed} {operator_class})'
    )


def _is_pgvector(vector_engine) -> bool:
    # Lazy, guarded import, as in postgres_graph_provenance_columns: the pgvector adapter
    # needs the postgres extra, and this migration runs for every vector backend.
    try:
        from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
            PGVectorAdapter,
        )
    except ImportError:
        return False

    return isinstance(vector_engine, PGVectorAdapter)


async def migrate(context: MigrationContext) -> None:
    if not _is_pgvector(context.vector_engine):
        return
    async with context.vector_engine.engine.begin() as conn:
        collections = (await conn.execute(text(_COLLECTIONS_SQL))).all()
        for table_name, dimensions in collections:
            statement = _frozen_index_statement(table_name, dimensions)
            if statement is not None:
                await conn.execute(text(statement))


async def downgrade(context: MigrationContext) -> None:
    if not _is_pgvector(context.vector_engine):
        return
    async with context.vector_engine.engine.begin() as conn:
        collections = (await conn.execute(text(_COLLECTIONS_SQL))).all()
        for table_name, _dimensions in collections:
            await conn.execute(text(f'DROP INDEX IF EXISTS "{table_name}_vector_hnsw"'))
