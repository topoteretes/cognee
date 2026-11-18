"""Move a Postgres graph's source refs from array columns into two ref tables.

A Postgres graph kept the source refs of a node or an edge in four ``varchar[]``
columns on ``graph_node`` and ``graph_edge`` (``source_ref_keys``,
``source_dataset_ids``, ``source_run_ids``, ``source_run_refs``; see
``postgres_graph_provenance_columns``). A node shared by many documents had its row
rewritten with a longer list on every attach. A ref is now a row of its own, in
``graph_node_source_ref`` or ``graph_edge_source_ref``. This migration creates the
two tables, copies every ref in its stored order and then drops the columns.

It acts on every graph engine whose ``engine`` is a Postgres one and whose
``graph_node``, resolved through the connection's search path (so a graph in a
dataset schema is found too), still has ``source_ref_keys``. The adapter class is
not looked at: whatever stores its graph in these tables has to be converted. Any
other graph is left alone, including a converted one and one whose tables do not
exist yet. The SQL is a frozen snapshot, like every migration: nothing is imported
from the adapter.

The work is sized for a large graph on a small server:

* the rows are copied one batch of ``_BATCH_ROWS`` graph_node (then graph_edge) rows
  at a time, walking the primary key, each batch in its own transaction;
* all copying is ``INSERT ... SELECT`` inside Postgres; only batch bounds and row
  counts cross to this process;
* the run of each ref comes from one pass over the batch's run refs, not from a
  search through a node's run refs for every one of its keys;
* the ref tables' lookup indexes are dropped before the copy and built once after
  it, so a copied row updates only its primary key. Nothing reads the ref tables
  meanwhile: the adapter refuses a graph that still has the columns.

The columns are dropped last, in one transaction for both tables and together with
the index builds, because their presence is the only mark of a graph that is not
converted yet. A run that stops anywhere before that is repeated from the start by
the next run: a ref already copied is skipped (``ON CONFLICT DO NOTHING``), so the
result is the same.

``downgrade`` is the reverse with the same shape: add the columns and their GIN
indexes, fill the arrays from the ref rows batch by batch, drop the two ref tables
last. It fills only nodes and edges that have ref rows, so the arrays of a graph
whose upgrade stopped half way are kept, and filling a row twice gives the same
arrays. Stamps and descriptions kept on node ref rows have no place in the columns
and are lost.

The runner holds its cross-process migration lock (``runner.migration_lock``)
around every upgrade and downgrade, so this takes no lock of its own. Like every
migration it expects that no older release writes to the graph while it runs: a
ref such a writer adds to a row that was already copied is not carried over.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from cognee.modules.migrations.migration import MigrationContext
from cognee.shared.logging_utils import get_logger

logger = get_logger("migration.postgres_graph_source_ref_tables")

# graph_node or graph_edge rows handled per transaction.
_BATCH_ROWS = 1000
# Progress is logged once per this many batches, and at the end.
_LOG_EVERY_BATCHES = 100

_LEGACY_COLUMNS = ("source_ref_keys", "source_dataset_ids", "source_run_ids", "source_run_refs")
# The three columns that had a GIN index (see ``postgres_graph_provenance_columns``).
_INDEXED_LEGACY_COLUMNS = ("source_ref_keys", "source_dataset_ids", "source_run_ids")


@dataclass(frozen=True)
class _Artifact:
    """One graph table, its ref table, and the primary key that links them."""

    table: str
    ref_table: str
    key: tuple[str, ...]  # primary key of ``table``
    ref_key: tuple[str, ...]  # the same key as ``ref_table`` names it
    index_prefix: str


_NODES = _Artifact("graph_node", "graph_node_source_ref", ("id",), ("node_id",), "node")
_EDGES = _Artifact(
    "graph_edge",
    "graph_edge_source_ref",
    ("source_id", "target_id", "relationship_name"),
    ("source_id", "target_id", "relationship_name"),
    "edge",
)
_ARTIFACTS = (_NODES, _EDGES)

# ``{schema}`` is the quoted schema graph_node lives in, so the ref tables are created
# next to it whatever the search path puts first.
_CREATE_REF_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.graph_node_source_ref (
        node_id text NOT NULL REFERENCES {schema}.graph_node (id) ON DELETE CASCADE,
        source_ref_key text NOT NULL,
        dataset_id text NOT NULL,
        pipeline_run_id text,
        position bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
        source_pipeline text,
        source_task text,
        source_node_set text,
        source_user text,
        source_content_hash text,
        description text,
        PRIMARY KEY (node_id, source_ref_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.graph_edge_source_ref (
        source_id text NOT NULL,
        target_id text NOT NULL,
        relationship_name text NOT NULL,
        source_ref_key text NOT NULL,
        dataset_id text NOT NULL,
        pipeline_run_id text,
        position bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
        PRIMARY KEY (source_id, target_id, relationship_name, source_ref_key),
        FOREIGN KEY (source_id, target_id, relationship_name)
            REFERENCES {schema}.graph_edge (source_id, target_id, relationship_name)
            ON DELETE CASCADE
    )
    """,
)

# The lookup indexes of the two ref tables: (index, table, column). The copy runs
# without them and they are built once after it. A row inserted while they exist
# updates all three, at a place its key or ids decide, not at the end.
_REF_LOOKUP_INDEXES = tuple(
    (f"idx_{prefix}_source_ref_{name}", f"graph_{prefix}_source_ref", column)
    for prefix in ("node", "edge")
    for name, column in (
        ("key", "source_ref_key"),
        ("dataset", "dataset_id"),
        ("run", "pipeline_run_id"),
    )
)

# The quoted schema of a table found through the search path; no row when it is absent.
_SCHEMA_OF_TABLE = """
    SELECT quote_ident(namespace.nspname)
      FROM pg_class AS relation
      JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
     WHERE relation.oid = to_regclass('{table}')
"""
_AND_HAS_LEGACY_COLUMNS = """
       AND EXISTS (
               SELECT 1 FROM pg_attribute AS attribute
                WHERE attribute.attrelid = relation.oid
                  AND attribute.attname = 'source_ref_keys'
                  AND NOT attribute.attisdropped
           )
"""

# The last key of the next batch: at most ``:batch_rows`` primary-key entries are read.
_NEXT_BATCH_BOUND = """
    SELECT {key}
      FROM (
            SELECT {key}
              FROM {schema}.{table}
             {after}
             ORDER BY {key}
             LIMIT :batch_rows
           ) AS batch
     ORDER BY {key_descending}
     LIMIT 1
"""

# The formats read here are frozen: they are what the earlier layout stored.
#   source ref key   source_ref:v<n>:<dataset id>:<data id>[:<chunk id>]
#   source run ref   source_run_ref:v1:<pipeline run id>:<source ref key>
# ``entry`` is every key of the batch with its place in its array, and every run ref
# of the batch taken apart. Grouped by node (or edge) and key, a key gets its place,
# which becomes the order of the rows, and the first run that names it; a key no run
# ref names was attached without a run, and a run ref for a key the node does not
# have is dropped. Keys and run refs are grouped, never joined: a join can end up
# comparing every key of a node with every one of its run refs, whatever the planner
# expects of the arrays, and a node with many refs cannot afford that.
# Ids pass through ``::uuid::text`` to be stored in the canonical form the adapter
# writes (lookups by dataset and run compare text), and a malformed id fails the batch.
_COPY_REFS = """
    INSERT INTO {schema}.{ref_table} ({ref_key}, source_ref_key, dataset_id, pipeline_run_id)
    SELECT {ref_columns}, ref.source_ref_key,
           split_part(ref.source_ref_key, ':', 3)::uuid::text,
           ref.pipeline_run_id
      FROM (
            SELECT {key}, source_ref_key,
                   min(position) FILTER (WHERE is_key) AS position,
                   (array_agg(pipeline_run_id ORDER BY position) FILTER (WHERE NOT is_key))[1]
                       AS pipeline_run_id
              FROM (
                    SELECT {artifact_columns}, item.source_ref_key, item.position,
                           true AS is_key, NULL::text AS pipeline_run_id
                      FROM {schema}.{table} AS artifact
                     CROSS JOIN LATERAL unnest(artifact.source_ref_keys)
                                WITH ORDINALITY AS item(source_ref_key, position)
                     WHERE {artifact_range}
                    UNION ALL
                    SELECT {artifact_columns},
                           substring(item.run_ref FROM '^[^:]*:[^:]*:[^:]*:(.*)$'),
                           item.position, false,
                           split_part(item.run_ref, ':', 3)::uuid::text
                      FROM {schema}.{table} AS artifact
                     CROSS JOIN LATERAL unnest(artifact.source_run_refs)
                                WITH ORDINALITY AS item(run_ref, position)
                     WHERE {artifact_range}
                   ) AS entry
             GROUP BY {key}, source_ref_key
            HAVING bool_or(is_key)
           ) AS ref
     ORDER BY {ref_columns}, ref.position
    ON CONFLICT DO NOTHING
"""

# The four arrays of every node or edge of the batch that has ref rows: the keys in
# ``position`` order, the dataset ids and run ids once each in the order they first
# appear, and a run ref for every ref that has a run.
_FILL_ARRAYS = """
    UPDATE {schema}.{table} AS artifact
       SET source_ref_keys = filled.source_ref_keys,
           source_dataset_ids = filled.source_dataset_ids,
           source_run_ids = filled.source_run_ids,
           source_run_refs = filled.source_run_refs
      FROM (
            SELECT {ref_key},
                   array_agg(source_ref_key ORDER BY position) AS source_ref_keys,
                   array_agg(dataset_id ORDER BY position) FILTER (WHERE first_of_dataset)
                       AS source_dataset_ids,
                   COALESCE(
                       array_agg(pipeline_run_id ORDER BY position) FILTER (WHERE first_of_run),
                       '{{}}'
                   ) AS source_run_ids,
                   COALESCE(
                       array_agg(
                           'source_run_ref:v1:' || pipeline_run_id || ':' || source_ref_key
                           ORDER BY position
                       ) FILTER (WHERE pipeline_run_id IS NOT NULL),
                       '{{}}'
                   ) AS source_run_refs
              FROM (
                    SELECT {ref_key}, source_ref_key, dataset_id, pipeline_run_id, position,
                           row_number() OVER (
                               PARTITION BY {ref_key}, dataset_id ORDER BY position
                           ) = 1 AS first_of_dataset,
                           pipeline_run_id IS NOT NULL AND row_number() OVER (
                               PARTITION BY {ref_key}, pipeline_run_id ORDER BY position
                           ) = 1 AS first_of_run
                      FROM {schema}.{ref_table} AS ref
                     WHERE {ref_range}
                   ) AS ref
             GROUP BY {ref_key}
           ) AS filled
     WHERE ({artifact_columns}) = ({filled_columns})
"""


def _columns(alias: str, columns: tuple[str, ...]) -> str:
    return ", ".join(f"{alias}.{column}" for column in columns)


def _parameters(prefix: str, count: int) -> str:
    return ", ".join(f":{prefix}_{index}" for index in range(count))


def _bind(prefix: str, values: tuple) -> dict:
    return {f"{prefix}_{index}": value for index, value in enumerate(values)}


def _batch_range(alias: str, columns: tuple[str, ...], after: tuple | None) -> str:
    """The rows of one batch: past the previous batch's last key, up to this one's.

    Row values are compared whole, so the edges' three-column key walks its
    primary-key index the same way the nodes' single column does.
    """
    row = f"({_columns(alias, columns)})"
    upper = f"{row} <= ({_parameters('upper', len(columns))})"
    if after is None:
        return upper
    return f"{row} > ({_parameters('after', len(columns))}) AND {upper}"


def _next_batch_bound(schema: str, artifact: _Artifact, after: tuple | None) -> str:
    key = ", ".join(artifact.key)
    return _NEXT_BATCH_BOUND.format(
        schema=schema,
        table=artifact.table,
        key=key,
        key_descending=", ".join(f"{column} DESC" for column in artifact.key),
        after=""
        if after is None
        else f"WHERE ({key}) > ({_parameters('after', len(artifact.key))})",
    )


def _copy_refs(schema: str, artifact: _Artifact, after: tuple | None) -> str:
    return _COPY_REFS.format(
        schema=schema,
        table=artifact.table,
        ref_table=artifact.ref_table,
        key=", ".join(artifact.key),
        ref_key=", ".join(artifact.ref_key),
        artifact_columns=_columns("artifact", artifact.key),
        ref_columns=_columns("ref", artifact.key),
        artifact_range=_batch_range("artifact", artifact.key, after),
    )


def _fill_arrays(schema: str, artifact: _Artifact, after: tuple | None) -> str:
    return _FILL_ARRAYS.format(
        schema=schema,
        table=artifact.table,
        ref_table=artifact.ref_table,
        ref_key=", ".join(artifact.ref_key),
        artifact_columns=_columns("artifact", artifact.key),
        filled_columns=_columns("filled", artifact.ref_key),
        ref_range=_batch_range("ref", artifact.ref_key, after),
    )


def _postgres_engine(graph_engine):
    """The graph's SQLAlchemy engine when the graph is stored in Postgres, else None."""
    engine = getattr(graph_engine, "engine", None)
    if getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None
    return engine


async def _schema_of(connection, table: str, with_legacy_columns: bool = False) -> str | None:
    """The quoted schema ``table`` is found in, or None when there is no such table."""
    statement = _SCHEMA_OF_TABLE.format(table=table)
    if with_legacy_columns:
        statement += _AND_HAS_LEGACY_COLUMNS
    return (await connection.execute(text(statement))).scalar()


async def _run_in_batches(connection, schema: str, artifact: _Artifact, statement_for) -> None:
    """Run ``statement_for(schema, artifact, after)`` once per batch of ``artifact.table``.

    Each batch is one transaction, so an interrupted run keeps the batches it finished.
    """
    after = None
    batches = rows = 0
    while True:
        bound = (
            await connection.execute(
                text(_next_batch_bound(schema, artifact, after)),
                {"batch_rows": _BATCH_ROWS, **_bind("after", after or ())},
            )
        ).first()
        if bound is None:
            break
        result = await connection.execute(
            text(statement_for(schema, artifact, after)),
            {**_bind("after", after or ()), **_bind("upper", tuple(bound))},
        )
        await connection.commit()
        after = tuple(bound)
        batches += 1
        rows += result.rowcount
        if batches % _LOG_EVERY_BATCHES == 0:
            logger.info(
                "%s: %d batches done, %d rows written so far", artifact.table, batches, rows
            )
    await connection.commit()
    logger.info("%s: finished, %d batches, %d rows written", artifact.table, batches, rows)


async def migrate(context: MigrationContext) -> None:
    engine = _postgres_engine(context.graph_engine)
    if engine is None:
        return
    async with engine.connect() as connection:
        schema = await _schema_of(connection, "graph_node", with_legacy_columns=True)
        if schema is None:
            return
        for statement in _CREATE_REF_TABLES:
            await connection.execute(text(statement.format(schema=schema)))
        # The adapter creates the ref tables with their lookup indexes before the
        # chain runs, and an earlier run of this migration may have built them.
        for index, _table, _column in _REF_LOOKUP_INDEXES:
            await connection.execute(text(f"DROP INDEX IF EXISTS {schema}.{index}"))
        await connection.commit()

        for artifact in _ARTIFACTS:
            await _run_in_batches(connection, schema, artifact, _copy_refs)

        # One transaction: a graph without the columns always has its lookup indexes.
        for index, table, column in _REF_LOOKUP_INDEXES:
            await connection.execute(
                text(f"CREATE INDEX IF NOT EXISTS {index} ON {schema}.{table} ({column})")
            )
        for artifact in _ARTIFACTS:
            drops = ", ".join(f"DROP COLUMN IF EXISTS {column}" for column in _LEGACY_COLUMNS)
            await connection.execute(text(f"ALTER TABLE {schema}.{artifact.table} {drops}"))
        await connection.commit()


async def downgrade(context: MigrationContext) -> None:
    engine = _postgres_engine(context.graph_engine)
    if engine is None:
        return
    async with engine.connect() as connection:
        schema = await _schema_of(connection, "graph_node_source_ref")
        if schema is None:
            return
        for artifact in _ARTIFACTS:
            table = f"{schema}.{artifact.table}"
            for column in _LEGACY_COLUMNS:
                await connection.execute(
                    text(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} "
                        "varchar[] NOT NULL DEFAULT '{}'"
                    )
                )
            for column in _INDEXED_LEGACY_COLUMNS:
                await connection.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS idx_{artifact.index_prefix}_{column} "
                        f"ON {table} USING gin ({column})"
                    )
                )
        await connection.commit()

        for artifact in _ARTIFACTS:
            await _run_in_batches(connection, schema, artifact, _fill_arrays)

        await connection.execute(
            text(f"DROP TABLE IF EXISTS {schema}.{_EDGES.ref_table}, {schema}.{_NODES.ref_table}")
        )
        await connection.commit()
