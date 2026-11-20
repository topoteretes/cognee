"""Source refs of the Postgres graph: the ref tables, their migration, the node write rule.

Requires a running Postgres instance. Connection defaults:
  DB_HOST=localhost, DB_PORT=5432, DB_USERNAME=cognee,
  DB_PASSWORD=cognee, DB_NAME=cognee_db

Every test but one works in a schema of its own, which it drops again. The one
that covers the default search path replaces the graph tables of ``public``.
"""

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.exceptions import PostgresGraphLayoutError
from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    make_chunk_source_ref_key,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.databases.provenance.source_ref_state import (
    ProvenanceColumns,
    provenance_after_attach,
)
from cognee.modules.engine.models import Entity
from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions import postgres_graph_provenance_columns
from cognee.modules.migrations.versions import postgres_graph_source_ref_tables as migration

pytestmark = pytest.mark.asyncio

_GRAPH_TABLES = (
    "graph_edge_source_ref",
    "graph_node_source_ref",
    "graph_edge",
    "graph_node",
    "graph_metadata",
)

# The layout before the ref tables, as the releases that had it created it. Frozen
# here on purpose: these tests are about graphs those releases left behind.
_OLD_PROVENANCE_COLUMNS = """
    source_ref_keys varchar[] NOT NULL DEFAULT '{}',
    source_dataset_ids varchar[] NOT NULL DEFAULT '{}',
    source_run_ids varchar[] NOT NULL DEFAULT '{}',
    source_run_refs varchar[] NOT NULL DEFAULT '{}',
"""
_OLD_LAYOUT = (
    f"""
    CREATE TABLE graph_node (
        id varchar NOT NULL PRIMARY KEY,
        name varchar,
        type varchar,
        properties jsonb,
        {_OLD_PROVENANCE_COLUMNS}
        created_at timestamptz DEFAULT now(),
        updated_at timestamptz DEFAULT now()
    )
    """,
    f"""
    CREATE TABLE graph_edge (
        source_id varchar NOT NULL REFERENCES graph_node (id) ON DELETE CASCADE,
        target_id varchar NOT NULL REFERENCES graph_node (id) ON DELETE CASCADE,
        relationship_name varchar NOT NULL,
        properties jsonb,
        {_OLD_PROVENANCE_COLUMNS}
        created_at timestamptz DEFAULT now(),
        updated_at timestamptz DEFAULT now(),
        PRIMARY KEY (source_id, target_id, relationship_name)
    )
    """,
    "CREATE TABLE graph_metadata (key varchar NOT NULL PRIMARY KEY, value varchar NOT NULL)",
    "CREATE INDEX idx_edge_source ON graph_edge (source_id)",
    "CREATE INDEX idx_edge_target ON graph_edge (target_id)",
    "CREATE INDEX idx_node_type ON graph_node (type)",
    "CREATE INDEX idx_edge_source_cover ON graph_edge (source_id)"
    + " INCLUDE (target_id, relationship_name)",
    "CREATE INDEX idx_edge_target_cover ON graph_edge (target_id)"
    + " INCLUDE (source_id, relationship_name)",
    "CREATE INDEX idx_node_source_ref_keys ON graph_node USING gin (source_ref_keys)",
    "CREATE INDEX idx_node_source_dataset_ids ON graph_node USING gin (source_dataset_ids)",
    "CREATE INDEX idx_node_source_run_ids ON graph_node USING gin (source_run_ids)",
    "CREATE INDEX idx_edge_source_ref_keys ON graph_edge USING gin (source_ref_keys)",
    "CREATE INDEX idx_edge_source_dataset_ids ON graph_edge USING gin (source_dataset_ids)",
    "CREATE INDEX idx_edge_source_run_ids ON graph_edge USING gin (source_run_ids)",
)
_OLD_GIN_INDEXES = {
    "idx_node_source_ref_keys",
    "idx_node_source_dataset_ids",
    "idx_node_source_run_ids",
    "idx_edge_source_ref_keys",
    "idx_edge_source_dataset_ids",
    "idx_edge_source_run_ids",
}
# What the adapter looks refs up by; the primary keys are not among them.
_REF_LOOKUP_INDEXES = {
    "idx_node_source_ref_key",
    "idx_node_source_ref_dataset",
    "idx_node_source_ref_run",
    "idx_edge_source_ref_key",
    "idx_edge_source_ref_dataset",
    "idx_edge_source_ref_run",
}

DATASETS = [UUID(int=1000 + n) for n in range(3)]
RUNS = [UUID(int=2000 + n) for n in range(3)]


def _url() -> str:
    username = os.environ.get("DB_USERNAME", "cognee")
    password = os.environ.get("DB_PASSWORD", "cognee")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    database = os.environ.get("DB_NAME", "cognee_db")
    return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"


def _key(dataset: int, data: int, chunk: int | None = None) -> str:
    """A document-scoped source ref key, or a chunk-scoped one when ``chunk`` is given."""
    if chunk is None:
        return make_source_ref_key(DATASETS[dataset], UUID(int=3000 + data))
    return make_chunk_source_ref_key(
        DATASETS[dataset], UUID(int=3000 + data), UUID(int=4000 + chunk)
    )


def _old_columns(attaches: list[tuple[str, UUID | None]]) -> ProvenanceColumns:
    """What the earlier layout held after these (key, run) attaches, in this order."""
    columns = ProvenanceColumns([], [], [], [])
    for key, run in attaches:
        columns = provenance_after_attach(
            columns.source_ref_keys, columns.source_run_refs, [key], run and str(run)
        )
    return columns


# A node with far more refs than a batch has rows: document and chunk keys of three
# datasets, attached under three runs or under none. Two keys are attached a second
# time, which must neither move them nor change the run they were first attached with.
_MANY = [(_key(n % 3, n, None if n % 4 else n), RUNS[n % 3] if n % 5 else None) for n in range(25)]
_MANY += [(_MANY[3][0], RUNS[1]), (_MANY[5][0], RUNS[2])]

NODE_ATTACHES: dict[str, list[tuple[str, UUID | None]]] = {
    "many": _MANY,
    "one-run": [(_key(0, 100), RUNS[0])],
    "no-run": [(_key(1, 101), None), (_key(1, 102), None)],
    "mixed": [(_key(2, 103), RUNS[2]), (_key(0, 104), None), (_key(2, 105), RUNS[1])],
    "bare": [],
    "bare-too": [],
    "last": [(_key(0, 106), RUNS[1]), (_key(0, 100), RUNS[1])],
    " odd'id": [(_key(1, 107), RUNS[2])],
    "": [(_key(2, 108), None)],
}
EDGE_ATTACHES: dict[EdgeIdentity, list[tuple[str, UUID | None]]] = {
    EdgeIdentity("many", "one-run", "mentions"): _MANY[:9],
    EdgeIdentity("many", "one-run", "contains"): [(_key(0, 100), RUNS[0]), (_key(1, 101), None)],
    EdgeIdentity("many", "many", "is_a"): [],
    EdgeIdentity("mixed", "bare", "is_a"): [(_key(2, 103), None), (_key(2, 105), RUNS[1])],
    EdgeIdentity("no-run", "bare", "is_a"): [],
    EdgeIdentity("one-run", "last", "made_from"): [(_key(0, 106), RUNS[1])],
    EdgeIdentity(" odd'id", "", "made_from"): [(_key(1, 107), RUNS[2])],
}
NODE_COLUMNS = {node_id: _old_columns(attaches) for node_id, attaches in NODE_ATTACHES.items()}
EDGE_COLUMNS = {edge: _old_columns(attaches) for edge, attaches in EDGE_ATTACHES.items()}


def _edge_parameters(edge: EdgeIdentity) -> dict:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relationship_name": edge.relationship_name,
    }


async def _build_old_graph(engine) -> None:
    """A graph in the earlier layout holding NODE_ATTACHES and EDGE_ATTACHES."""
    arrays = (
        "CAST(:source_ref_keys AS varchar[]), CAST(:source_dataset_ids AS varchar[]), "
        "CAST(:source_run_ids AS varchar[]), CAST(:source_run_refs AS varchar[])"
    )
    async with engine.begin() as connection:
        for statement in _OLD_LAYOUT:
            await connection.execute(text(statement))
        for node_id, columns in NODE_COLUMNS.items():
            await connection.execute(
                text(f"""
                    INSERT INTO graph_node (id, name, type, properties,
                        source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs)
                    VALUES (:id, :id, 'Thing', '{{"weight": 1}}', {arrays})
                """),
                {"id": node_id, **columns._asdict()},
            )
        for edge, columns in EDGE_COLUMNS.items():
            await connection.execute(
                text(f"""
                    INSERT INTO graph_edge (source_id, target_id, relationship_name, properties,
                        source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs)
                    VALUES (:source_id, :target_id, :relationship_name, '{{}}', {arrays})
                """),
                {**_edge_parameters(edge), **columns._asdict()},
            )


def _context(graph_engine) -> MigrationContext:
    return MigrationContext(graph_engine=graph_engine, vector_engine=None)


@pytest_asyncio.fixture
async def graph():
    """An adapter on an empty schema of its own; the graph tables are not created."""
    schema = f"srt_{uuid4().hex[:16]}"
    admin = create_async_engine(_url())
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    adapter = PostgresDemoAdapter(_url(), schema=schema)
    try:
        yield adapter
    finally:
        await adapter.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest_asyncio.fixture
async def old_graph(graph):
    """An adapter on a graph in the earlier layout."""
    await _build_old_graph(graph.engine)
    return graph


@pytest_asyncio.fixture
async def current_graph(graph):
    """An adapter on an empty graph in the current layout."""
    await graph.initialize()
    return graph


async def _rows(engine, statement: str, parameters: dict | None = None) -> list[tuple]:
    async with engine.connect() as connection:
        return [tuple(row) for row in await connection.execute(text(statement), parameters or {})]


async def _columns_of(engine, table: str) -> list[str]:
    rows = await _rows(
        engine,
        """
        SELECT attname FROM pg_attribute
         WHERE attrelid = to_regclass(:table) AND attnum > 0 AND NOT attisdropped
         ORDER BY attnum
        """,
        {"table": table},
    )
    return [row[0] for row in rows]


async def _index_names(engine) -> set[str]:
    rows = await _rows(
        engine, "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
    )
    return {row[0] for row in rows}


async def _has_old_layout(engine) -> bool:
    return "source_ref_keys" in await _columns_of(engine, "graph_node")


async def _layout(engine) -> dict:
    """Everything the catalog says about the graph tables, without the schema's name."""
    (schema,) = (await _rows(engine, "SELECT current_schema()"))[0]

    def anonymous(value: str | None) -> str | None:
        return value and value.replace(f"{schema}.", "").replace(f'"{schema}".', "")

    columns = await _rows(
        engine,
        """
        SELECT relation.relname, attribute.attname, format_type(atttypid, atttypmod),
               attnotnull, attidentity::text, pg_get_expr(adbin, adrelid)
          FROM pg_attribute AS attribute
          JOIN pg_class AS relation ON relation.oid = attribute.attrelid
          LEFT JOIN pg_attrdef ON adrelid = attribute.attrelid AND adnum = attribute.attnum
         WHERE relation.relnamespace = current_schema()::regnamespace
           AND relation.relkind = 'r' AND attnum > 0 AND NOT attisdropped
         ORDER BY relation.relname, attnum
        """,
    )
    constraints = await _rows(
        engine,
        """
        SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid)
          FROM pg_constraint
         WHERE connamespace = current_schema()::regnamespace AND contype IN ('p', 'f')
         ORDER BY 1, 2
        """,
    )
    indexes = await _rows(
        engine,
        """
        SELECT indexname, indexdef FROM pg_indexes
         WHERE schemaname = current_schema() ORDER BY indexname
        """,
    )
    return {
        "columns": columns,
        "constraints": [tuple(anonymous(value) for value in row) for row in constraints],
        "indexes": [tuple(anonymous(value) for value in row) for row in indexes],
    }


async def _ref_rows(engine) -> dict:
    """Both ref tables whole, each node's or edge's refs in attach order."""
    return {
        "nodes": await _rows(
            engine,
            """
            SELECT node_id, source_ref_key, dataset_id, pipeline_run_id, source_pipeline,
                   source_task, source_node_set, source_user, source_content_hash, description
              FROM graph_node_source_ref ORDER BY node_id, position
            """,
        ),
        "edges": await _rows(
            engine,
            """
            SELECT source_id, target_id, relationship_name, source_ref_key, dataset_id,
                   pipeline_run_id
              FROM graph_edge_source_ref
             ORDER BY source_id, target_id, relationship_name, position
            """,
        ),
    }


def _refs_of_dataset(columns_by_artifact: dict, dataset_id: UUID) -> dict:
    """What ``find_*_source_refs_by_dataset`` answered from the earlier layout."""
    out = {}
    for artifact, columns in columns_by_artifact.items():
        owned = [
            key
            for key in columns.source_ref_keys
            if get_dataset_id_from_source_ref_key(key) == dataset_id
        ]
        if owned:
            out[artifact] = owned
    return out


def _refs_of_run(columns_by_artifact: dict, run_id: UUID) -> dict:
    """What ``find_*_source_refs_by_pipeline_run`` answered from the earlier layout."""
    out = {}
    for artifact, columns in columns_by_artifact.items():
        attached = [
            get_source_ref_key_from_source_run_ref(run_ref)
            for run_ref in columns.source_run_refs
            if get_pipeline_run_id_from_source_run_ref(run_ref) == run_id
        ]
        if attached:
            out[artifact] = attached
    return out


async def _assert_adapter_reports_the_old_refs(adapter) -> None:
    """Every ownership answer of the adapter equals what the earlier layout stored."""
    nodes = await adapter.get_node_delete_data(list(NODE_COLUMNS))
    assert set(nodes) == set(NODE_COLUMNS)
    for node_id, columns in NODE_COLUMNS.items():
        snapshot = nodes[node_id]
        assert snapshot.source_ref_keys == columns.source_ref_keys, node_id
        assert snapshot.source_dataset_ids == columns.source_dataset_ids, node_id
        assert snapshot.source_run_ids == columns.source_run_ids, node_id
        assert snapshot.source_run_refs == columns.source_run_refs, node_id
        assert snapshot.node_properties == {
            "id": node_id,
            "name": node_id,
            "type": "Thing",
            "weight": 1,
        }

    edges = await adapter.get_edge_delete_data(list(EDGE_COLUMNS))
    assert set(edges) == set(EDGE_COLUMNS)
    for edge, columns in EDGE_COLUMNS.items():
        snapshot = edges[edge]
        assert snapshot.source_ref_keys == columns.source_ref_keys, edge
        assert snapshot.source_dataset_ids == columns.source_dataset_ids, edge
        assert snapshot.source_run_ids == columns.source_run_ids, edge
        assert snapshot.source_run_refs == columns.source_run_refs, edge

    every_key = {key for columns in NODE_COLUMNS.values() for key in columns.source_ref_keys}
    for key in every_key | {_key(0, 999)}:
        assert set(await adapter.find_nodes_by_source_ref(key)) == {
            node_id for node_id, columns in NODE_COLUMNS.items() if key in columns.source_ref_keys
        }
        assert set(await adapter.find_edges_by_source_ref(key)) == {
            edge for edge, columns in EDGE_COLUMNS.items() if key in columns.source_ref_keys
        }
    for dataset_id in [*DATASETS, UUID(int=1)]:
        assert await adapter.find_node_source_refs_by_dataset(str(dataset_id)) == (
            _refs_of_dataset(NODE_COLUMNS, dataset_id)
        )
        assert await adapter.find_edge_source_refs_by_dataset(str(dataset_id)) == (
            _refs_of_dataset(EDGE_COLUMNS, dataset_id)
        )
    for run_id in [*RUNS, UUID(int=2)]:
        assert await adapter.find_node_source_refs_by_pipeline_run(str(run_id)) == (
            _refs_of_run(NODE_COLUMNS, run_id)
        )
        assert await adapter.find_edge_source_refs_by_pipeline_run(str(run_id)) == (
            _refs_of_run(EDGE_COLUMNS, run_id)
        )


async def _assert_old_layout_holds_the_refs(engine, filled_from_ref_rows: bool = True) -> None:
    """The array columns hold every ref again, as a release with that layout reads them.

    Arrays a downgrade filled list each dataset and run id once, in the order its
    first ref appears. Arrays it left as they were list them sorted, as written.
    """
    stored = {
        row[0]: row[1:]
        for row in await _rows(
            engine,
            """
            SELECT id, source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs
              FROM graph_node
            """,
        )
    }
    stored.update(
        {
            EdgeIdentity(*row[:3]): row[3:]
            for row in await _rows(
                engine,
                """
                SELECT source_id, target_id, relationship_name,
                       source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs
                  FROM graph_edge
                """,
            )
        }
    )
    expected = {**NODE_COLUMNS, **EDGE_COLUMNS}
    assert set(stored) == set(expected)
    for artifact, columns in expected.items():
        keys, dataset_ids, run_ids, run_refs = stored[artifact]
        assert keys == columns.source_ref_keys, artifact
        assert run_refs == columns.source_run_refs, artifact
        assert sorted(dataset_ids) == columns.source_dataset_ids, artifact
        assert sorted(run_ids) == columns.source_run_ids, artifact
        if filled_from_ref_rows:
            assert dataset_ids == list(
                dict.fromkeys(str(get_dataset_id_from_source_ref_key(key)) for key in keys)
            ), artifact
            assert run_ids == list(
                dict.fromkeys(str(get_pipeline_run_id_from_source_run_ref(ref)) for ref in run_refs)
            ), artifact


@contextmanager
def _statements(engine):
    """Collect every statement the engine sends, with its parameters."""
    seen: list[tuple[str, tuple]] = []

    def record(connection, cursor, statement, parameters, context, executemany):
        seen.append((" ".join(statement.split()), tuple(parameters or ())))

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)


class _Interrupted(Exception):
    """Stands in for whatever ends a migration run half way."""


@contextmanager
def _interrupting(engine, start: str, occurrence: int):
    """Make the ``occurrence``-th statement that starts with ``start`` fail."""
    count = 0

    def interrupt(connection, cursor, statement, parameters, context, executemany):
        nonlocal count
        if statement.lstrip().startswith(start):
            count += 1
            if count == occurrence:
                raise _Interrupted(f"{start} #{occurrence}")

    event.listen(engine.sync_engine, "before_cursor_execute", interrupt)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", interrupt)


# -- The migration -----------------------------------------------------------------


async def test_old_layout_graph_is_converted_with_the_same_refs_order_and_runs(old_graph):
    await migration.migrate(_context(old_graph))

    await _assert_adapter_reports_the_old_refs(old_graph)
    kept = ["properties", "created_at", "updated_at"]
    assert await _columns_of(old_graph.engine, "graph_node") == ["id", "name", "type", *kept]
    assert await _columns_of(old_graph.engine, "graph_edge") == [
        "source_id",
        "target_id",
        "relationship_name",
        *kept,
    ]
    assert not _OLD_GIN_INDEXES & await _index_names(old_graph.engine)
    # A converted ref has no stamps and no description: the earlier layout kept none.
    assert all(row[4:] == (None,) * 6 for row in (await _ref_rows(old_graph.engine))["nodes"])


async def test_key_named_by_several_run_refs_takes_the_first_one(graph):
    """The run of a ref is the first run ref that names its key, in the stored order."""
    keys = [_key(0, 1), _key(1, 2), _key(2, 3)]
    run_refs = [
        make_source_run_ref(RUNS[2], keys[1]),
        make_source_run_ref(RUNS[0], keys[0]),
        make_source_run_ref(RUNS[1], keys[0]),
        make_source_run_ref(RUNS[1], keys[1]),
    ]
    async with graph.engine.begin() as connection:
        for statement in _OLD_LAYOUT:
            await connection.execute(text(statement))
        await connection.execute(
            text("""
                INSERT INTO graph_node (id, source_ref_keys, source_run_refs)
                VALUES ('n', CAST(:keys AS varchar[]), CAST(:run_refs AS varchar[]))
            """),
            {"keys": keys, "run_refs": run_refs},
        )

    await migration.migrate(_context(graph))

    assert [row[:2] for row in await _node_refs(graph, "n")] == [
        (keys[0], str(RUNS[0])),
        (keys[1], str(RUNS[2])),
        (keys[2], None),
    ]


async def _build_old_node(engine, keys: list[str], run_refs: list[str]) -> None:
    """A graph in the earlier layout with one node "n" holding these keys and run refs."""
    async with engine.begin() as connection:
        for statement in _OLD_LAYOUT:
            await connection.execute(text(statement))
        await connection.execute(
            text("""
                INSERT INTO graph_node (id, source_ref_keys, source_run_refs)
                VALUES ('n', CAST(:keys AS varchar[]), CAST(:run_refs AS varchar[]))
            """),
            {"keys": keys, "run_refs": run_refs},
        )


async def test_ids_are_copied_in_the_form_the_adapter_looks_them_up_by(graph):
    """An id that parses but is not in canonical form is stored canonical."""
    dataset, run = str(DATASETS[0]), str(RUNS[0])
    key = make_source_ref_key(DATASETS[0], UUID(int=3001)).replace(dataset, dataset.upper())
    run_ref = make_source_run_ref(RUNS[0], key).replace(f":{run}:", f":{run.upper()}:")
    await _build_old_node(graph.engine, [key], [run_ref])

    await migration.migrate(_context(graph))

    assert await _rows(
        graph.engine, "SELECT dataset_id, pipeline_run_id FROM graph_node_source_ref"
    ) == [(dataset, run)]
    assert await graph.find_node_source_refs_by_dataset(dataset) == {"n": [key]}
    assert await graph.find_node_source_refs_by_pipeline_run(run) == {"n": [key]}


async def test_malformed_id_stops_the_migration(graph):
    """A key whose dataset id does not parse fails the copy instead of being stored."""
    await _build_old_node(graph.engine, [f"source_ref:v1:not-a-uuid:{UUID(int=3001)}"], [])

    with pytest.raises(DBAPIError, match="invalid input syntax for type uuid"):
        await migration.migrate(_context(graph))

    assert await _has_old_layout(graph.engine)


async def test_only_the_keys_of_a_node_become_rows_each_once(graph):
    """A run ref for a key the node does not have is dropped; a repeated key keeps its first place."""
    keys = [_key(0, 1), _key(1, 2), _key(0, 1)]
    run_refs = [make_source_run_ref(RUNS[0], _key(2, 9)), make_source_run_ref(RUNS[1], keys[1])]
    await _build_old_node(graph.engine, keys, run_refs)

    await migration.migrate(_context(graph))

    assert [row[:2] for row in await _node_refs(graph, "n")] == [
        (keys[0], None),
        (keys[1], str(RUNS[1])),
    ]


async def test_node_with_many_refs_is_copied_with_its_order_and_runs(graph):
    """Every key of a node shared by many documents keeps its place and its own run."""
    keys = [_key(n % 3, n, n) for n in range(3000)]
    runs = [RUNS[n % 3] for n in range(3000)]
    # The run refs are stored in another order than the keys they name.
    run_refs = [make_source_run_ref(run, key) for run, key in zip(runs, keys)][::-1]
    await _build_old_node(graph.engine, keys, run_refs)

    await migration.migrate(_context(graph))

    assert [row[:2] for row in await _node_refs(graph, "n")] == [
        (key, str(run)) for key, run in zip(keys, runs)
    ]


async def test_migration_creates_the_tables_the_adapter_creates(old_graph):
    """The frozen DDL and the adapter's schema must describe the same tables."""
    await migration.migrate(_context(old_graph))
    migrated = await _layout(old_graph.engine)

    schema = f"srt_{uuid4().hex[:16]}"
    async with old_graph.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    fresh = PostgresDemoAdapter(_url(), schema=schema)
    try:
        await fresh.initialize()
        created = await _layout(fresh.engine)
    finally:
        await fresh.close()
        async with old_graph.engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))

    assert migrated == created
    assert [row[1:] for row in created["columns"] if row[0] == "graph_node_source_ref"] == [
        ("node_id", "text", True, "", None),
        ("source_ref_key", "text", True, "", None),
        ("dataset_id", "text", True, "", None),
        ("pipeline_run_id", "text", False, "", None),
        ("position", "bigint", True, "a", None),
        ("source_pipeline", "text", False, "", None),
        ("source_task", "text", False, "", None),
        ("source_node_set", "text", False, "", None),
        ("source_user", "text", False, "", None),
        ("source_content_hash", "text", False, "", None),
        ("description", "text", False, "", None),
    ]
    assert [row[1:] for row in created["columns"] if row[0] == "graph_edge_source_ref"] == [
        ("source_id", "text", True, "", None),
        ("target_id", "text", True, "", None),
        ("relationship_name", "text", True, "", None),
        ("source_ref_key", "text", True, "", None),
        ("dataset_id", "text", True, "", None),
        ("pipeline_run_id", "text", False, "", None),
        ("position", "bigint", True, "a", None),
    ]
    ref_constraints = {row[2] for row in created["constraints"] if "source_ref" in row[0]}
    assert ref_constraints == {
        "PRIMARY KEY (node_id, source_ref_key)",
        "FOREIGN KEY (node_id) REFERENCES graph_node(id) ON DELETE CASCADE",
        "PRIMARY KEY (source_id, target_id, relationship_name, source_ref_key)",
        "FOREIGN KEY (source_id, target_id, relationship_name) REFERENCES"
        + " graph_edge(source_id, target_id, relationship_name) ON DELETE CASCADE",
    }
    assert {name for name, _definition in created["indexes"]} == {
        "graph_node_pkey",
        "graph_edge_pkey",
        "graph_metadata_pkey",
        "graph_node_source_ref_pkey",
        "graph_edge_source_ref_pkey",
        "idx_edge_source",
        "idx_edge_target",
        "idx_node_type",
        "idx_edge_source_cover",
        "idx_edge_target_cover",
        "idx_node_source_ref_key",
        "idx_node_source_ref_dataset",
        "idx_node_source_ref_run",
        "idx_edge_source_ref_key",
        "idx_edge_source_ref_dataset",
        "idx_edge_source_ref_run",
    }


async def test_migrating_twice_changes_nothing(old_graph):
    await migration.migrate(_context(old_graph))
    layout = await _layout(old_graph.engine)
    rows = await _rows(old_graph.engine, "SELECT * FROM graph_node_source_ref ORDER BY position")

    with _statements(old_graph.engine) as statements:
        await migration.migrate(_context(old_graph))

    # A converted graph costs one catalog lookup.
    assert len(statements) == 1
    assert await _layout(old_graph.engine) == layout
    assert rows == await _rows(
        old_graph.engine, "SELECT * FROM graph_node_source_ref ORDER BY position"
    )


async def test_downgrade_then_upgrade_round_trips(old_graph):
    await migration.migrate(_context(old_graph))

    await migration.downgrade(_context(old_graph))

    await _assert_old_layout_holds_the_refs(old_graph.engine)
    assert _OLD_GIN_INDEXES <= await _index_names(old_graph.engine)
    assert await _rows(
        old_graph.engine,
        "SELECT to_regclass('graph_node_source_ref'), to_regclass('graph_edge_source_ref')",
    ) == [(None, None)]
    # Nothing is left to downgrade.
    with _statements(old_graph.engine) as statements:
        await migration.downgrade(_context(old_graph))
    assert len(statements) == 1

    await migration.migrate(_context(old_graph))

    await _assert_adapter_reports_the_old_refs(old_graph)


async def test_graph_behind_another_schema_on_the_search_path_is_converted(old_graph):
    """Any engine on these tables is converted, and the ref tables join graph_node."""
    first = f"srt_{uuid4().hex[:16]}"
    async with old_graph.engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{first}"'))
    engine = create_async_engine(
        _url(),
        connect_args={"server_settings": {"search_path": f"{first},{old_graph.schema}"}},
    )
    try:
        # Not an adapter: only its engine says where the graph is.
        await migration.migrate(_context(SimpleNamespace(engine=engine)))

        assert await _rows(
            engine,
            """
            SELECT relname, relnamespace::regnamespace::text FROM pg_class
             WHERE relname IN ('graph_node_source_ref', 'graph_edge_source_ref')
               AND relnamespace::regnamespace::text IN (:first, :graph)
             ORDER BY relname
            """,
            {"first": first, "graph": old_graph.schema},
        ) == [
            ("graph_edge_source_ref", old_graph.schema),
            ("graph_node_source_ref", old_graph.schema),
        ]
        await _assert_adapter_reports_the_old_refs(old_graph)
    finally:
        await engine.dispose()
        async with old_graph.engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{first}" CASCADE'))


async def test_graph_in_the_default_schema_is_converted():
    """Without a schema of its own the graph is found where the search path puts it."""
    adapter = PostgresDemoAdapter(_url())
    drop = "DROP TABLE IF EXISTS " + ", ".join(_GRAPH_TABLES) + " CASCADE"
    try:
        async with adapter.engine.begin() as connection:
            await connection.execute(text(drop))
        await _build_old_graph(adapter.engine)

        await migration.migrate(_context(adapter))

        await _assert_adapter_reports_the_old_refs(adapter)
    finally:
        async with adapter.engine.begin() as connection:
            await connection.execute(text(drop))
        await adapter.close()


async def test_running_the_chain_again_keeps_a_converted_graph_converted():
    """The earlier columns migration must not add the arrays back to a converted graph.

    It looks for the graph in ``public`` only, so this test uses the default schema.
    """
    adapter = PostgresDemoAdapter(_url())
    drop = "DROP TABLE IF EXISTS " + ", ".join(_GRAPH_TABLES) + " CASCADE"
    try:
        async with adapter.engine.begin() as connection:
            await connection.execute(text(drop))
        await adapter.initialize()

        await postgres_graph_provenance_columns.migrate(_context(adapter))

        assert "source_ref_keys" not in await _columns_of(adapter.engine, "graph_node")
        await adapter.add_nodes([("n", {"name": "n", "type": "Thing"})], source_ref_key=_key(0, 1))
        assert [row[0] for row in await _node_refs(adapter, "n")] == [_key(0, 1)]
    finally:
        async with adapter.engine.begin() as connection:
            await connection.execute(text(drop))
        await adapter.close()


async def test_graph_without_the_old_columns_is_left_alone(current_graph):
    keys = [_key(0, 1), _key(1, 2)]
    await current_graph.add_nodes(
        [("n", {"name": "n", "type": "Entity", "description": "kept", "source_task": "task"})],
        source_ref_key=keys[0],
        pipeline_run_id=str(RUNS[0]),
    )
    await current_graph.attach_node_source_refs(["n"], [keys[1]])
    await current_graph.add_edges([("n", "n", "self", {})], source_ref_key=keys[0])
    layout = await _layout(current_graph.engine)
    rows = await _ref_rows(current_graph.engine)

    with _statements(current_graph.engine) as statements:
        await migration.migrate(_context(current_graph))

    assert len(statements) == 1
    assert await _layout(current_graph.engine) == layout
    assert await _ref_rows(current_graph.engine) == rows


async def test_graph_that_is_not_in_these_postgres_tables_is_left_alone(graph):
    class _NotPostgres:
        """An engine that fails on any use, as a graph stored elsewhere must never see one."""

        dialect = SimpleNamespace(name="sqlite")

        def connect(self):
            raise AssertionError("a graph outside Postgres was opened")

    for graph_engine in (
        object(),
        SimpleNamespace(engine=None),
        SimpleNamespace(engine=_NotPostgres()),
    ):
        await migration.migrate(_context(graph_engine))
        await migration.downgrade(_context(graph_engine))

    # Postgres, but the graph tables do not exist yet: nothing is created for it.
    await migration.migrate(_context(graph))
    await migration.downgrade(_context(graph))
    assert (
        await _rows(
            graph.engine, "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
        )
        == []
    )


# -- The migration in batches --------------------------------------------------------


async def test_conversion_spanning_many_batches_keeps_refs_order_and_runs(old_graph, monkeypatch):
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    assert len(NODE_COLUMNS["many"].source_ref_keys) > 10 * migration._BATCH_ROWS

    with _statements(old_graph.engine) as statements:
        await migration.migrate(_context(old_graph))

    copies = [statement for statement, _ in statements if statement.startswith("INSERT INTO")]
    assert len(copies) == -(-len(NODE_COLUMNS) // 2) + -(-len(EDGE_COLUMNS) // 2)
    await _assert_adapter_reports_the_old_refs(old_graph)

    await migration.downgrade(_context(old_graph))
    await _assert_old_layout_holds_the_refs(old_graph.engine)


def _batch_ends(keys: list[tuple], size: int) -> list[tuple]:
    """The last key of every batch when ``keys`` are walked ``size`` at a time."""
    return [keys[min(start + size, len(keys)) - 1] for start in range(0, len(keys), size)]


async def test_only_the_schema_steps_touch_a_whole_table(old_graph, monkeypatch):
    """Every statement that reads or writes graph rows is held to one batch."""
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    keys = {
        "graph_node": await _rows(old_graph.engine, "SELECT id FROM graph_node ORDER BY id"),
        "graph_edge": await _rows(
            old_graph.engine,
            """
            SELECT source_id, target_id, relationship_name FROM graph_edge
             ORDER BY source_id, target_id, relationship_name
            """,
        ),
    }

    with _statements(old_graph.engine) as statements:
        await migration.migrate(_context(old_graph))
        await migration.downgrade(_context(old_graph))

    schema_steps = ("CREATE TABLE", "CREATE INDEX", "DROP INDEX", "ALTER TABLE", "DROP TABLE")
    row_statements = [
        (statement, parameters)
        for statement, parameters in statements
        if not statement.startswith(schema_steps) and not statement.startswith("SELECT quote_ident")
    ]
    assert {statement.split()[0] for statement, _ in row_statements} == {
        "SELECT",
        "INSERT",
        "UPDATE",
    }
    batch_ends = {("INSERT", table): [] for table in keys}
    batch_ends.update({("UPDATE", table): [] for table in keys})
    for statement, parameters in row_statements:
        if statement.startswith("SELECT"):
            # Finding where a batch ends reads at most one batch of key entries.
            assert " LIMIT $" in statement and 2 in parameters, statement
            continue
        # A write is bounded wherever it reads graph rows: the keys and the run refs
        # of a copy, the ref rows of a fill.
        verb = statement.split()[0]
        assert statement.count(" <= ($") == (2 if verb == "INSERT" else 1), statement
        table = "graph_node" if "graph_node_source_ref" in statement else "graph_edge"
        batch_ends[(verb, table)].append(parameters[-len(keys[table][0]) :])
    # And the bound is the batch's: every write covers the next two rows, no more.
    for (verb, table), ends in batch_ends.items():
        assert ends == _batch_ends(keys[table], 2), (verb, table)


async def test_finished_batches_are_kept_when_a_later_one_fails(old_graph, monkeypatch):
    """Each batch is a transaction of its own, in both directions."""
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    node_ids = [
        row[0] for row in await _rows(old_graph.engine, "SELECT id FROM graph_node ORDER BY id")
    ]
    # The nodes of the first two batches that have refs.
    done = {node_id for node_id in node_ids[:4] if NODE_COLUMNS[node_id].source_ref_keys}
    assert done

    with _interrupting(old_graph.engine, "INSERT INTO", 3), pytest.raises(_Interrupted):
        await migration.migrate(_context(old_graph))
    assert {row[0] for row in (await _ref_rows(old_graph.engine))["nodes"]} == done

    await migration.migrate(_context(old_graph))
    with _interrupting(old_graph.engine, "UPDATE", 3), pytest.raises(_Interrupted):
        await migration.downgrade(_context(old_graph))
    filled = await _rows(
        old_graph.engine, "SELECT id FROM graph_node WHERE source_ref_keys <> '{}'"
    )
    assert {row[0] for row in filled} == done


@pytest.mark.parametrize(
    "start, occurrence",
    [
        pytest.param("INSERT INTO", 2, id="among-the-node-batches"),
        pytest.param("INSERT INTO", 6, id="at-the-first-edge-batch"),
        pytest.param("INSERT INTO", 8, id="among-the-edge-batches"),
        pytest.param("CREATE INDEX", 1, id="before-the-lookup-indexes-are-built"),
        pytest.param("CREATE INDEX", 4, id="among-the-index-builds"),
        pytest.param("ALTER TABLE", 1, id="before-the-columns-are-dropped"),
        pytest.param("ALTER TABLE", 2, id="between-the-two-drops"),
    ],
)
async def test_interrupted_upgrade_converges_when_run_again(
    old_graph, monkeypatch, start, occurrence
):
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    # Five node batches, then four edge batches.
    assert (len(NODE_COLUMNS), len(EDGE_COLUMNS)) == (9, 7)

    with _interrupting(old_graph.engine, start, occurrence), pytest.raises(_Interrupted):
        await migration.migrate(_context(old_graph))

    # Still marked as not converted, whatever was copied so far.
    assert await _has_old_layout(old_graph.engine)
    assert "source_ref_keys" in await _columns_of(old_graph.engine, "graph_edge")
    with pytest.raises(PostgresGraphLayoutError):
        await old_graph.get_node_delete_data(["many"])

    await migration.migrate(_context(old_graph))

    await _assert_adapter_reports_the_old_refs(old_graph)
    assert not await _has_old_layout(old_graph.engine)
    assert _REF_LOOKUP_INDEXES <= await _index_names(old_graph.engine)


async def test_lookup_indexes_are_dropped_for_the_copy_and_built_after_it(old_graph):
    """The copy runs without the lookup indexes, even when they were there before it."""
    # As the runner leaves a graph before the chain: the adapter has created the
    # ref tables, with their indexes, next to the earlier layout.
    await old_graph.initialize()
    assert _REF_LOOKUP_INDEXES <= await _index_names(old_graph.engine)

    with _statements(old_graph.engine) as statements:
        await migration.migrate(_context(old_graph))

    steps = [statement for statement, _ in statements]
    copies = [at for at, step in enumerate(steps) if step.startswith("INSERT INTO")]
    drops = [at for at, step in enumerate(steps) if step.startswith("DROP INDEX")]
    builds = [at for at, step in enumerate(steps) if step.startswith("CREATE INDEX")]
    columns_dropped = [at for at, step in enumerate(steps) if step.startswith("ALTER TABLE")]
    assert (len(drops), len(builds)) == (6, 6)
    assert max(drops) < min(copies)
    assert max(copies) < min(builds) and max(builds) < min(columns_dropped)
    assert _REF_LOOKUP_INDEXES <= await _index_names(old_graph.engine)
    await _assert_adapter_reports_the_old_refs(old_graph)


async def test_run_stopped_during_the_copy_leaves_no_lookup_index(old_graph):
    """Until the next run finishes, the graph is refused, so nothing misses them."""
    await old_graph.initialize()

    with _interrupting(old_graph.engine, "INSERT INTO", 1), pytest.raises(_Interrupted):
        await migration.migrate(_context(old_graph))

    names = await _index_names(old_graph.engine)
    assert not _REF_LOOKUP_INDEXES & names
    assert {"graph_node_source_ref_pkey", "graph_edge_source_ref_pkey"} <= names
    with pytest.raises(PostgresGraphLayoutError):
        await old_graph.find_node_source_refs_by_dataset(str(DATASETS[0]))

    # The next start: a new adapter is initialized on the ref tables as they were
    # left, without their indexes, and the chain runs again.
    restarted = PostgresDemoAdapter(_url(), schema=old_graph.schema)
    try:
        await restarted.initialize()
        await migration.migrate(_context(restarted))

        assert _REF_LOOKUP_INDEXES <= await _index_names(restarted.engine)
        await _assert_adapter_reports_the_old_refs(restarted)
    finally:
        await restarted.close()


async def test_interrupted_upgrade_gives_the_rows_of_an_uninterrupted_one(old_graph, monkeypatch):
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    with _interrupting(old_graph.engine, "INSERT INTO", 3), pytest.raises(_Interrupted):
        await migration.migrate(_context(old_graph))
    await migration.migrate(_context(old_graph))
    resumed = await _ref_rows(old_graph.engine)

    await migration.downgrade(_context(old_graph))
    await migration.migrate(_context(old_graph))

    assert resumed == await _ref_rows(old_graph.engine)


@pytest.mark.parametrize(
    "start, occurrence",
    [
        pytest.param("CREATE INDEX", 4, id="while-the-columns-are-added"),
        pytest.param("UPDATE", 2, id="among-the-node-batches"),
        pytest.param("UPDATE", 6, id="at-the-first-edge-batch"),
        pytest.param("DROP TABLE", 1, id="before-the-ref-tables-are-dropped"),
    ],
)
@pytest.mark.parametrize("then", ["downgrade", "migrate"])
async def test_interrupted_downgrade_converges_either_way(
    current_graph, monkeypatch, start, occurrence, then
):
    """A downgrade that stopped is finished by another one, and undone by an upgrade."""
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    await _write_current_graph(current_graph)
    async with current_graph.engine.begin() as connection:
        await connection.execute(
            text("UPDATE graph_node_source_ref SET source_task = 'task', description = 'kept'")
        )
    before = await _ref_rows(current_graph.engine)

    with _interrupting(current_graph.engine, start, occurrence), pytest.raises(_Interrupted):
        await migration.downgrade(_context(current_graph))
    # The ref rows are the truth until the very end of a downgrade.
    assert await _ref_rows(current_graph.engine) == before

    if then == "downgrade":
        await migration.downgrade(_context(current_graph))
        await _assert_old_layout_holds_the_refs(current_graph.engine)
    else:
        await migration.migrate(_context(current_graph))
        # Nothing was copied back: every row is the one that was there, stamps included.
        assert await _ref_rows(current_graph.engine) == before
        await _assert_adapter_reports_the_old_refs(current_graph)


@pytest.mark.parametrize("occurrence", [1, 3, 6])
async def test_downgrade_after_an_interrupted_upgrade_keeps_every_ref(
    old_graph, monkeypatch, occurrence
):
    """Only some refs have rows then, so the arrays of the others must stay as they are."""
    monkeypatch.setattr(migration, "_BATCH_ROWS", 2)
    with _interrupting(old_graph.engine, "INSERT INTO", occurrence), pytest.raises(_Interrupted):
        await migration.migrate(_context(old_graph))

    await migration.downgrade(_context(old_graph))

    await _assert_old_layout_holds_the_refs(old_graph.engine, filled_from_ref_rows=False)
    assert await _rows(old_graph.engine, "SELECT to_regclass('graph_node_source_ref')") == [(None,)]


async def _write_current_graph(adapter) -> None:
    """NODE_ATTACHES and EDGE_ATTACHES written through the adapter, in the current layout."""
    await adapter.add_nodes(
        [(node_id, {"name": node_id, "type": "Thing", "weight": 1}) for node_id in NODE_ATTACHES]
    )
    await adapter.add_edges(
        [(edge.source_id, edge.target_id, edge.relationship_name, {}) for edge in EDGE_ATTACHES]
    )
    for node_id, attaches in NODE_ATTACHES.items():
        for key, run in attaches:
            await adapter.attach_node_source_refs([node_id], [key], run and str(run))
    for edge, attaches in EDGE_ATTACHES.items():
        for key, run in attaches:
            await adapter.attach_edge_source_refs([edge], [key], run and str(run))


async def test_adapter_reports_what_the_earlier_layout_reported(current_graph):
    """The same attaches give the same answers, whichever layout they were written in."""
    await _write_current_graph(current_graph)

    await _assert_adapter_reports_the_old_refs(current_graph)


# -- A graph that is not converted yet ---------------------------------------------

_EDGE = EdgeIdentity("many", "one-run", "mentions")
_REFUSED = {
    "add_nodes": lambda adapter: adapter.add_nodes([("new", {"name": "new"})]),
    "add_node": lambda adapter: adapter.add_node("new", {"name": "new"}),
    "add_edges": lambda adapter: adapter.add_edges([("many", "bare", "new", {})]),
    "add_edge": lambda adapter: adapter.add_edge("many", "bare", "new"),
    "attach_node_source_refs": lambda adapter: adapter.attach_node_source_refs(
        ["bare"], [_key(0, 1)]
    ),
    "attach_edge_source_refs": lambda adapter: adapter.attach_edge_source_refs(
        [_EDGE], [_key(0, 1)]
    ),
    "remove_node_source_refs": lambda adapter: adapter.remove_node_source_refs(
        ["many"], [_key(0, 100)]
    ),
    "remove_edge_source_refs": lambda adapter: adapter.remove_edge_source_refs(
        [_EDGE], [_key(0, 100)]
    ),
    "get_node_delete_data": lambda adapter: adapter.get_node_delete_data(["many"]),
    "get_edge_delete_data": lambda adapter: adapter.get_edge_delete_data([_EDGE]),
    "find_nodes_by_source_ref": lambda adapter: adapter.find_nodes_by_source_ref(_key(0, 100)),
    "find_edges_by_source_ref": lambda adapter: adapter.find_edges_by_source_ref(_key(0, 100)),
    "find_node_source_refs_by_dataset": lambda adapter: adapter.find_node_source_refs_by_dataset(
        str(DATASETS[0])
    ),
    "find_edge_source_refs_by_dataset": lambda adapter: adapter.find_edge_source_refs_by_dataset(
        str(DATASETS[0])
    ),
    "find_node_source_refs_by_data": lambda adapter: adapter.find_node_source_refs_by_data(
        str(DATASETS[0]), str(uuid4())
    ),
    "find_edge_source_refs_by_data": lambda adapter: adapter.find_edge_source_refs_by_data(
        str(DATASETS[0]), str(uuid4())
    ),
    "find_node_source_refs_by_pipeline_run": lambda adapter: (
        adapter.find_node_source_refs_by_pipeline_run(str(RUNS[0]))
    ),
    "find_edge_source_refs_by_pipeline_run": lambda adapter: (
        adapter.find_edge_source_refs_by_pipeline_run(str(RUNS[0]))
    ),
}


async def test_unconverted_graph_is_refused_until_the_migration_ran(old_graph):
    # The migration runner initializes the adapter first, so this must not raise.
    await old_graph.initialize()
    before = await _rows(old_graph.engine, "SELECT * FROM graph_node ORDER BY id")

    for name, call in _REFUSED.items():
        with pytest.raises(PostgresGraphLayoutError) as refused:
            await call(old_graph)
        message = str(refused.value)
        assert "earlier layout" in message, name
        assert "cognee.run_migrations()" in message and "cognee-cli upgrade" in message
        assert "upgrade to cognee 1.6.8 first" in message
    assert await _rows(old_graph.engine, "SELECT * FROM graph_node ORDER BY id") == before
    assert (await old_graph.get_node("many"))["name"] == "many"

    await migration.migrate(_context(old_graph))

    # The same adapter object, no restart.
    await _assert_adapter_reports_the_old_refs(old_graph)
    await old_graph.add_nodes([("new", {"name": "new"})], source_ref_key=_key(0, 1))
    assert await old_graph.find_nodes_by_source_ref(_key(0, 1)) == ["new"]


async def test_current_graph_is_looked_up_once(current_graph):
    with _statements(current_graph.engine) as statements:
        await current_graph.add_nodes([("n", {"name": "n"})])
        await current_graph.get_node_delete_data(["n"])
        await current_graph.find_nodes_by_source_ref(_key(0, 1))

    assert sum("pg_attribute" in statement for statement, _ in statements) == 1


# -- The node write rule -------------------------------------------------------------

_STAMP_COLUMNS = (
    "source_pipeline",
    "source_task",
    "source_node_set",
    "source_user",
    "source_content_hash",
)


def _stamps(tag: str) -> dict:
    return {column: f"{column}-{tag}" for column in _STAMP_COLUMNS}


async def _stored_node(adapter, node_id: str) -> tuple[str, dict]:
    """Where the node's row is and what it holds: a rewrite moves the row."""
    ((place, properties),) = await _rows(
        adapter.engine,
        "SELECT ctid::text, properties FROM graph_node WHERE id = :id",
        {"id": node_id},
    )
    return place, properties


async def _node_refs(adapter, node_id: str) -> list[tuple]:
    """(key, run, description, the five stamps) of each ref of the node, in attach order."""
    return await _rows(
        adapter.engine,
        f"""
        SELECT source_ref_key, pipeline_run_id, description, {", ".join(_STAMP_COLUMNS)}
          FROM graph_node_source_ref WHERE node_id = :id ORDER BY position
        """,
        {"id": node_id},
    )


async def test_entity_keeps_its_first_description_and_each_ref_its_own(current_graph):
    first = Entity(name="paris", description="A city.", **_stamps("first"))
    second = Entity(
        name="paris", description="The same city in other words.", created_at=1, **_stamps("second")
    )
    third = Entity(name="paris", description="A third view.")
    assert first.id == second.id == third.id
    node_id = str(first.id)

    await current_graph.add_nodes([first], source_ref_key=_key(0, 1), pipeline_run_id=str(RUNS[0]))
    place, stored = await _stored_node(current_graph, node_id)
    await current_graph.add_nodes([second], source_ref_key=_key(0, 2), pipeline_run_id=str(RUNS[1]))
    await current_graph.add_nodes([third], source_ref_key=_key(1, 3))

    assert await _stored_node(current_graph, node_id) == (place, stored)
    node = await current_graph.get_node(node_id)
    assert node["description"] == "A city."
    assert {column: node[column] for column in _STAMP_COLUMNS} == _stamps("first")
    assert await _node_refs(current_graph, node_id) == [
        (_key(0, 1), str(RUNS[0]), "A city.", *_stamps("first").values()),
        (_key(0, 2), str(RUNS[1]), "The same city in other words.", *_stamps("second").values()),
        (_key(1, 3), None, "A third view.", None, None, None, None, None),
    ]


async def test_stamps_alone_do_not_rewrite_a_node_of_any_type(current_graph):
    chunk = {"name": "", "type": "DocumentChunk", "text": "body", "created_at": 1}
    await current_graph.add_nodes(
        [("chunk", {**chunk, **_stamps("first")})], source_ref_key=_key(0, 1)
    )
    place, stored = await _stored_node(current_graph, "chunk")

    await current_graph.add_nodes(
        [("chunk", {**chunk, **_stamps("second"), "created_at": 2, "updated_at": 3})],
        source_ref_key=_key(0, 2),
    )
    # A write that carries only some stamps, and none of the timestamps.
    await current_graph.add_nodes(
        [("chunk", {"name": "", "type": "DocumentChunk", "text": "body", "source_task": "t"})],
        source_ref_key=_key(0, 3),
    )

    assert await _stored_node(current_graph, "chunk") == (place, stored)
    assert await _node_refs(current_graph, "chunk") == [
        (_key(0, 1), None, None, *_stamps("first").values()),
        (_key(0, 2), None, None, *_stamps("second").values()),
        (_key(0, 3), None, None, None, "t", None, None, None),
    ]


@pytest.mark.parametrize("empty", [{}, {"description": None}, {"description": ""}])
async def test_empty_description_gives_way_to_a_real_one_and_never_replaces_one(
    current_graph, empty
):
    entity = {"name": "oslo", "type": "Entity"}
    await current_graph.add_nodes([("oslo", {**entity, **empty})], source_ref_key=_key(0, 1))
    place, _ = await _stored_node(current_graph, "oslo")

    # Another empty value is no reason to rewrite.
    for other in ({}, {"description": None}, {"description": ""}):
        await current_graph.add_nodes([("oslo", {**entity, **other})])
    assert (await _stored_node(current_graph, "oslo"))[0] == place

    await current_graph.add_nodes(
        [("oslo", {**entity, "description": "A capital."})], source_ref_key=_key(0, 2)
    )
    rewritten, stored = await _stored_node(current_graph, "oslo")
    assert rewritten != place
    assert stored == {"description": "A capital."}

    await current_graph.add_nodes([("oslo", {**entity, **empty})], source_ref_key=_key(0, 3))
    assert await _stored_node(current_graph, "oslo") == (rewritten, stored)
    assert [row[2] for row in await _node_refs(current_graph, "oslo")] == [
        empty.get("description"),
        "A capital.",
        empty.get("description"),
    ]


async def test_any_other_difference_rewrites_the_node_with_the_incoming_properties(
    current_graph,
):
    entity = {"name": "rome", "type": "Entity", "weight": 1}
    await current_graph.add_nodes(
        [("rome", {**entity, "description": "First.", **_stamps("first")})]
    )
    place, _ = await _stored_node(current_graph, "rome")

    await current_graph.add_nodes(
        [("rome", {**entity, "weight": 2, "description": "Second.", **_stamps("second")})]
    )

    rewritten, stored = await _stored_node(current_graph, "rome")
    assert rewritten != place
    assert stored == {"weight": 2, "description": "Second.", **_stamps("second")}

    # The two columns outside the properties count as content too.
    for change in ({"name": "roma"}, {"type": "Place"}):
        before, _ = await _stored_node(current_graph, "rome")
        await current_graph.add_nodes([("rome", {**entity, "weight": 2, **change})])
        after, stored = await _stored_node(current_graph, "rome")
        assert after != before
        assert stored == {"weight": 2}
        entity = {**entity, **change}


async def test_description_and_relations_are_content_on_other_types(current_graph):
    summary = {"name": "s", "type": "TextSummary", "description": "First.", "relations": []}
    await current_graph.add_nodes([("s", summary)], source_ref_key=_key(0, 1))
    place, _ = await _stored_node(current_graph, "s")

    await current_graph.add_nodes(
        [("s", {**summary, "description": "Second."})], source_ref_key=_key(0, 2)
    )
    rewritten, stored = await _stored_node(current_graph, "s")
    assert rewritten != place
    assert stored["description"] == "Second."

    await current_graph.add_nodes([("s", {**summary, "description": "Second.", "relations": [1]})])
    assert (await _stored_node(current_graph, "s"))[0] != rewritten
    # Only an Entity's description is kept on its ref rows.
    assert [row[2] for row in await _node_refs(current_graph, "s")] == [None, None]


async def test_relations_of_an_entity_are_not_compared(current_graph):
    entity = {"name": "lima", "type": "Entity", "description": "A port."}
    await current_graph.add_nodes([("lima", {**entity, "relations": []})])
    place, stored = await _stored_node(current_graph, "lima")

    await current_graph.add_nodes([("lima", entity)])
    await current_graph.add_nodes([("lima", {**entity, "relations": [["near", "callao"]]})])

    assert await _stored_node(current_graph, "lima") == (place, stored)


async def test_ref_attached_without_a_node_write_has_no_stamps(current_graph):
    await current_graph.add_nodes(
        [("cairo", {"name": "cairo", "type": "Entity", "description": "D.", **_stamps("w")})]
    )

    await current_graph.attach_node_source_refs(
        ["cairo", "nobody"], [_key(0, 1), _key(0, 2)], str(RUNS[0])
    )

    assert await _node_refs(current_graph, "cairo") == [
        (_key(0, 1), str(RUNS[0]), None, None, None, None, None, None),
        (_key(0, 2), str(RUNS[0]), None, None, None, None, None, None),
    ]
    # A node that does not exist is skipped, as before.
    assert await _node_refs(current_graph, "nobody") == []


async def test_ref_that_is_already_there_keeps_its_run_stamps_and_description(current_graph):
    entity = {"name": "quito", "type": "Entity"}
    await current_graph.add_nodes(
        [("quito", {**entity, "description": "First.", **_stamps("first")})],
        source_ref_key=_key(0, 1),
        pipeline_run_id=str(RUNS[0]),
    )
    await current_graph.add_edges(
        [("quito", "quito", "self", {})], source_ref_key=_key(0, 1), pipeline_run_id=str(RUNS[0])
    )
    edge = EdgeIdentity("quito", "quito", "self")
    node_rows = await _rows(current_graph.engine, "SELECT * FROM graph_node_source_ref")
    edge_rows = await _rows(current_graph.engine, "SELECT * FROM graph_edge_source_ref")

    await current_graph.add_nodes(
        [("quito", {**entity, "description": "Second.", **_stamps("second")})],
        source_ref_key=_key(0, 1),
        pipeline_run_id=str(RUNS[1]),
    )
    await current_graph.attach_node_source_refs(["quito"], [_key(0, 1)], str(RUNS[2]))
    await current_graph.attach_node_source_refs(["quito"], [_key(0, 1)])
    await current_graph.add_edges(
        [("quito", "quito", "self", {"weight": 2})],
        source_ref_key=_key(0, 1),
        pipeline_run_id=str(RUNS[1]),
    )
    await current_graph.attach_edge_source_refs([edge], [_key(0, 1)], str(RUNS[2]))

    assert await _rows(current_graph.engine, "SELECT * FROM graph_node_source_ref") == node_rows
    assert await _rows(current_graph.engine, "SELECT * FROM graph_edge_source_ref") == edge_rows
    # An edge write is not compared: it replaces the edge as it always did.
    assert (await current_graph.get_edge_delete_data([edge]))[edge].edge_properties == {"weight": 2}
    snapshot = (await current_graph.get_node_delete_data(["quito"]))["quito"]
    assert snapshot.source_run_refs == [make_source_run_ref(RUNS[0], _key(0, 1))]


# -- Callers that replace a node on purpose ------------------------------------------


async def test_replace_rewrites_a_node_that_differs_only_in_its_description(current_graph):
    first = Entity(name="paris", description="A city.", **_stamps("first"))
    second = Entity(name="paris", description="An edited description.", **_stamps("second"))
    node_id = str(first.id)
    await current_graph.add_nodes([first], source_ref_key=_key(0, 1))
    await current_graph.add_nodes([second], source_ref_key=_key(0, 2))
    assert (await current_graph.get_node(node_id))["description"] == "A city."

    await current_graph.add_nodes([second], source_ref_key=_key(0, 1), replace=True)

    node = await current_graph.get_node(node_id)
    assert node["description"] == "An edited description."
    assert {column: node[column] for column in _STAMP_COLUMNS} == _stamps("second")
    # The ref rows follow the usual rule: a ref already there keeps its own write.
    assert [row[2] for row in await _node_refs(current_graph, node_id)] == [
        "A city.",
        "An edited description.",
    ]


async def _ingested_entity(adapter, name: str, description: str) -> Entity:
    """An entity as cognify writes it: stamped, with its document's ref."""
    entity = Entity(
        name=name, description=description, source_pipeline="cognify", source_task="extract"
    )
    await adapter.add_nodes([entity], source_ref_key=_key(0, 1), pipeline_run_id=str(RUNS[0]))
    return entity


async def test_merging_duplicate_entities_stores_the_merged_description(current_graph, monkeypatch):
    from cognee.tasks.memify import consolidate_entities

    city = await _ingested_entity(current_graph, "new york city", "The largest US city.")
    short = await _ingested_entity(current_graph, "nyc", "Short for New York City.")
    members = [
        {
            "id": str(entity.id),
            "name": entity.name,
            "description": entity.description,
            "created_at": index,
            "props": await current_graph.get_node(str(entity.id)),
        }
        for index, entity in enumerate((city, short))
    ]

    class _VectorEngine:
        async def delete_data_points(self, collection, ids):
            pass

    async def graph_engine():
        return current_graph

    monkeypatch.setattr(consolidate_entities, "get_graph_engine", graph_engine)
    monkeypatch.setattr(consolidate_entities, "get_vector_engine", lambda: _VectorEngine())

    await consolidate_entities.merge_entity_duplicates({"clusters": [members], "edges": []})

    assert (await current_graph.get_node(str(city.id)))["description"] == (
        "The largest US city. Short for New York City."
    )
    assert await current_graph.get_node(str(short.id)) is None


async def test_rewriting_entity_descriptions_stores_the_new_description(current_graph, monkeypatch):
    """The pipeline's own storage task, run on the rewritten entity it produces."""
    from cognee.memify_pipelines import consolidate_entity_descriptions as pipeline
    from cognee.tasks.memify.consolidate_entity_descriptions.rewrite_entities import (
        build_entity,
    )
    from cognee.tasks.storage.add_data_points import add_data_points

    entity = await _ingested_entity(current_graph, "paris", "A city.")
    enrichment_tasks = []

    async def run_memify_tasks(**kwargs):
        enrichment_tasks.extend(kwargs["enrichment_tasks"])

    async def graph_engine():
        return current_graph

    monkeypatch.setattr(pipeline, "run_memify_tasks", run_memify_tasks)
    # graph_only=True looks the engine up in this module (the package exports a function
    # of the same name, so the module is taken from sys.modules).
    graph_engines = sys.modules["cognee.infrastructure.databases.graph.get_graph_engine"]
    monkeypatch.setattr(graph_engines, "get_graph_engine", graph_engine)
    with pytest.warns(DeprecationWarning):
        await pipeline.consolidate_entity_descriptions_pipeline()
    (store,) = [task for task in enrichment_tasks if task.executable is add_data_points]

    stored = await current_graph.get_node(str(entity.id))
    rewritten = build_entity(stored, [], "The capital of France.")
    await store.run([rewritten], graph_only=True)

    assert (await current_graph.get_node(str(entity.id)))["description"] == (
        "The capital of France."
    )


# -- Deleting and removing -----------------------------------------------------------


async def test_ref_rows_go_with_their_node_or_edge(current_graph):
    await current_graph.add_nodes(
        [("a", {"name": "a"}), ("b", {"name": "b"}), ("c", {"name": "c"})],
        source_ref_key=_key(0, 1),
    )
    await current_graph.add_edges(
        [("a", "b", "r", {}), ("b", "c", "r", {}), ("c", "a", "r", {})],
        source_ref_key=_key(0, 1),
    )
    await current_graph.attach_edge_source_refs([EdgeIdentity("b", "c", "r")], [_key(0, 2)])

    await current_graph.delete_edge_triples([EdgeIdentity("b", "c", "r")])
    assert [row[:3] for row in (await _ref_rows(current_graph.engine))["edges"]] == [
        ("a", "b", "r"),
        ("c", "a", "r"),
    ]

    # Deleting a node takes its own refs and those of its edges.
    await current_graph.delete_nodes(["a"])
    rows = await _ref_rows(current_graph.engine)
    assert [row[0] for row in rows["nodes"]] == ["b", "c"]
    assert rows["edges"] == []
    assert sorted(await current_graph.find_nodes_by_source_ref(_key(0, 1))) == ["b", "c"]

    await current_graph.delete_graph()
    assert await _ref_rows(current_graph.engine) == {"nodes": [], "edges": []}


async def test_removing_refs_down_to_the_last_one(current_graph):
    keys = [_key(0, 1), _key(1, 2), _key(0, 3)]
    edge = EdgeIdentity("n", "n", "self")
    await current_graph.add_nodes([("n", {"name": "n"})])
    await current_graph.add_edges([("n", "n", "self", {})])
    await current_graph.attach_node_source_refs(["n"], keys, str(RUNS[0]))
    await current_graph.attach_edge_source_refs([edge], keys, str(RUNS[0]))

    await current_graph.remove_node_source_refs(["n", "nobody"], [keys[1], _key(2, 9)])
    await current_graph.remove_edge_source_refs([edge], [keys[1], _key(2, 9)])

    for snapshot in (
        (await current_graph.get_node_delete_data(["n"]))["n"],
        (await current_graph.get_edge_delete_data([edge]))[edge],
    ):
        assert snapshot.source_ref_keys == [keys[0], keys[2]]
        assert snapshot.source_dataset_ids == [str(DATASETS[0])]
        assert snapshot.source_run_ids == [str(RUNS[0])]
        assert snapshot.source_run_refs == [
            make_source_run_ref(RUNS[0], keys[0]),
            make_source_run_ref(RUNS[0], keys[2]),
        ]

    await current_graph.remove_node_source_refs(["n"], [keys[0], keys[2]])
    await current_graph.remove_edge_source_refs([edge], [keys[0], keys[2]])
    # Removing what is no longer there does nothing.
    await current_graph.remove_node_source_refs(["n"], keys)

    for snapshot in (
        (await current_graph.get_node_delete_data(["n"]))["n"],
        (await current_graph.get_edge_delete_data([edge]))[edge],
    ):
        assert snapshot.source_ref_keys == []
        assert snapshot.source_dataset_ids == []
        assert snapshot.source_run_ids == []
        assert snapshot.source_run_refs == []
    # The node and the edge themselves stay; whoever removed the last ref deletes them.
    assert await current_graph.has_node("n")
    assert await current_graph.has_edge("n", "n", "self")
    assert await current_graph.find_node_source_refs_by_dataset(str(DATASETS[0])) == {}
    assert await current_graph.find_edge_source_refs_by_pipeline_run(str(RUNS[0])) == {}

    # A ref attached again is a new one: it goes last and takes the new run.
    await current_graph.attach_node_source_refs(["n"], [keys[2], keys[0]], str(RUNS[1]))
    snapshot = (await current_graph.get_node_delete_data(["n"]))["n"]
    assert snapshot.source_ref_keys == [keys[2], keys[0]]
    assert snapshot.source_run_ids == [str(RUNS[1])]
