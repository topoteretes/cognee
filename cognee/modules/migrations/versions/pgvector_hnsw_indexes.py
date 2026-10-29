"""Add an HNSW index to every pgvector collection that was created without one.

New collections get the index from ``PGVectorAdapter.create_collection``. Collections
created by earlier releases have only a primary key, so every search over them is a full
scan that reads every stored vector: a 3072-dimensional collection of 12k rows is ~160 MB
of TOAST pages per query, 20 s when the pages are not cached. This builds the same index
for them.

Only the pgvector backend is affected; for every other vector backend this is a no-op.
The DDL is a frozen snapshot of ``hnsw_index_statement`` and ``hnsw_index_name`` (a
migration is immutable history, so it does not import the adapter's helpers). Statements
are idempotent (``IF NOT EXISTS``) and the store is read from the Postgres catalog, so an
empty store does nothing. Tables are found in ``current_schema()``, which is the dataset's
own schema in shared-database mode (the adapter pins ``search_path``) and ``public``
otherwise.

The index only makes searches faster; a collection without one is still searched
correctly, by a full scan. So each table is indexed in its own transaction (a build
blocks writes to that table only, not to the whole store), and a table whose build fails
(a ``statement_timeout``, a full disk) is logged and skipped rather than failing the
migration: a failed migration blocks every write to the dataset.
``PGVectorAdapter.run_migrations``, which the runner calls on every cognee version
change, builds any index that is still missing, so a skipped table is retried with the
next release. On a pgvector older
than 0.5.0 (no HNSW), or older than 0.7.0 for collections over 2,000 dimensions (no
halfvec), the collection is skipped the same way the adapter skips it.
"""

from __future__ import annotations

import hashlib
import re

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from cognee.modules.migrations.migration import MigrationContext
from cognee.shared.logging_utils import get_logger

logger = get_logger("migration.pgvector_hnsw_indexes")

_FROZEN_VECTOR_MAX_DIMENSIONS = 2000
_FROZEN_HALFVEC_MAX_DIMENSIONS = 4000
_FROZEN_HNSW_MIN_PGVECTOR = (0, 5, 0)
_FROZEN_HALFVEC_MIN_PGVECTOR = (0, 7, 0)
_FROZEN_MAX_IDENTIFIER_BYTES = 63

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

_PGVECTOR_VERSION_SQL = "SELECT extversion FROM pg_extension WHERE extname = 'vector'"

# An HNSW build is much slower once its graph outgrows maintenance_work_mem (12k
# 3072-dimensional rows on a 1 CU Neon compute: 22.6 s at its 67 MB default, 12.4 s at
# 256 MB). Raised for the build's own transaction only, and never lowered.
_BUILD_MAINTENANCE_WORK_MEM_KB = 256 * 1024
_RAISE_MAINTENANCE_WORK_MEM_SQL = """
SELECT set_config('maintenance_work_mem', :target, true)
FROM pg_settings
WHERE name = 'maintenance_work_mem' AND setting::bigint < :target_kb
"""


def _frozen_pgvector_version(version: str | None) -> tuple[int, ...] | None:
    parts = re.findall(r"\d+", version or "")
    return tuple(int(part) for part in parts[:3]) if parts else None


def _frozen_index_name(table_name: str) -> str:
    name = f"{table_name}_vector_hnsw"
    if len(name.encode()) <= _FROZEN_MAX_IDENTIFIER_BYTES:
        return name
    digest = hashlib.sha1(table_name.encode()).hexdigest()[:12]
    prefix = table_name.encode()[:40].decode(errors="ignore")
    return f"{prefix}_{digest}_hnsw"


def _frozen_index_statement(
    table_name: str, dimensions: int, pgvector_version: tuple[int, ...] | None
) -> str | None:
    if pgvector_version is None or pgvector_version < _FROZEN_HNSW_MIN_PGVECTOR:
        return None
    if dimensions <= 0:
        return None
    if dimensions <= _FROZEN_VECTOR_MAX_DIMENSIONS:
        indexed, operator_class = "vector", "vector_cosine_ops"
    elif (
        dimensions <= _FROZEN_HALFVEC_MAX_DIMENSIONS
        and pgvector_version >= _FROZEN_HALFVEC_MIN_PGVECTOR
    ):
        indexed, operator_class = f"(vector::halfvec({dimensions}))", "halfvec_cosine_ops"
    else:
        return None
    return (
        f'CREATE INDEX IF NOT EXISTS "{_frozen_index_name(table_name)}" '
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


async def _collections(engine) -> tuple[list[tuple[str, int]], tuple[int, ...] | None]:
    async with engine.connect() as conn:
        collections = [tuple(row) for row in (await conn.execute(text(_COLLECTIONS_SQL))).all()]
        version = (await conn.execute(text(_PGVECTOR_VERSION_SQL))).scalar_one_or_none()
    return collections, _frozen_pgvector_version(version)


async def migrate(context: MigrationContext) -> None:
    if not _is_pgvector(context.vector_engine):
        return
    engine = context.vector_engine.engine
    collections, pgvector_version = await _collections(engine)
    for table_name, dimensions in collections:
        statement = _frozen_index_statement(table_name, dimensions, pgvector_version)
        if statement is None:
            continue
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(_RAISE_MAINTENANCE_WORK_MEM_SQL),
                    {
                        "target": f"{_BUILD_MAINTENANCE_WORK_MEM_KB}kB",
                        "target_kb": _BUILD_MAINTENANCE_WORK_MEM_KB,
                    },
                )
                await conn.execute(text(statement))
        except SQLAlchemyError as error:
            logger.warning(
                "Could not build the HNSW index on %s (%s); it stays searchable by a full scan "
                "and is retried on the next cognee version change: %s",
                table_name,
                type(error).__name__,
                error,
            )


async def downgrade(context: MigrationContext) -> None:
    if not _is_pgvector(context.vector_engine):
        return
    engine = context.vector_engine.engine
    collections, _version = await _collections(engine)
    for table_name, _dimensions in collections:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP INDEX IF EXISTS "{_frozen_index_name(table_name)}"'))
