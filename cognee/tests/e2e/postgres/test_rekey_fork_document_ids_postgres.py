"""``rekey_fork_document_ids`` on a Postgres graph still in the source_ref_keys layout.

That layout is what the graph is in when this migration runs (the ref tables come
later in the chain), and the current adapter refuses it, so the migration does the
graph half in its own SQL. These tests build such a graph by hand and run that SQL.

Requires a running Postgres instance. Connection defaults:
  DB_HOST=localhost, DB_PORT=5432, DB_USERNAME=cognee,
  DB_PASSWORD=cognee, DB_NAME=cognee_db

Every test works in a schema of its own, which it drops again.
"""

import json
import os
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.provenance import (
    make_chunk_source_ref_key,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions import rekey_fork_document_ids as migration

pytestmark = pytest.mark.asyncio

# The layout before the ref tables, as the releases that had it created it.
_PROVENANCE = """
    source_ref_keys varchar[] NOT NULL DEFAULT '{}',
    source_dataset_ids varchar[] NOT NULL DEFAULT '{}',
    source_run_ids varchar[] NOT NULL DEFAULT '{}',
    source_run_refs varchar[] NOT NULL DEFAULT '{}',
"""
_OLD_LAYOUT = (
    f"""CREATE TABLE graph_node (
        id varchar NOT NULL PRIMARY KEY, name varchar, type varchar, properties jsonb,
        {_PROVENANCE}
        created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now())""",
    f"""CREATE TABLE graph_edge (
        source_id varchar NOT NULL REFERENCES graph_node (id) ON DELETE CASCADE,
        target_id varchar NOT NULL REFERENCES graph_node (id) ON DELETE CASCADE,
        relationship_name varchar NOT NULL, properties jsonb,
        {_PROVENANCE}
        created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
        PRIMARY KEY (source_id, target_id, relationship_name))""",
)

DATASET = UUID(int=10)
OTHER_DOCUMENT = UUID(int=11)
RUN = UUID(int=20)
LEGACY = str(UUID(int=30))
CANONICAL = str(UUID(int=31))
CHUNK = str(UUID(int=40))
SUMMARY = str(UUID(int=41))
ENTITY = str(UUID(int=42))
UNRELATED = str(UUID(int=43))


def _document_key(data_id: str) -> str:
    return make_source_ref_key(DATASET, UUID(data_id))


def _chunk_key(data_id: str) -> str:
    return make_chunk_source_ref_key(DATASET, UUID(data_id), UUID(CHUNK))


# node id -> (type, properties, source ref keys). Every key is attached under RUN.
NODES = {
    LEGACY: ("TextDocument", {"name": "doc"}, [_document_key(LEGACY)]),
    CHUNK: (
        "DocumentChunk",
        {"text": "Pierre met Natasha.", "document_id": LEGACY, "chunk_index": 0},
        [_document_key(LEGACY), _chunk_key(LEGACY)],
    ),
    SUMMARY: ("TextSummary", {"text": "A meeting."}, [_chunk_key(LEGACY)]),
    # Already carries the canonical key too: a run stopped half way. Shared with
    # another document, whose key must stay where it is.
    ENTITY: (
        "Entity",
        {"name": "pierre"},
        [
            make_source_ref_key(DATASET, OTHER_DOCUMENT),
            _document_key(LEGACY),
            _document_key(CANONICAL),
        ],
    ),
    UNRELATED: ("Entity", {"name": "natasha"}, [make_source_ref_key(DATASET, OTHER_DOCUMENT)]),
}
# (source, target, relationship) -> source ref keys
EDGES = {
    (CHUNK, LEGACY, "is_part_of"): [_document_key(LEGACY)],
    (SUMMARY, CHUNK, "made_from"): [_chunk_key(LEGACY)],
    (CHUNK, ENTITY, "contains"): [_document_key(LEGACY), _chunk_key(LEGACY)],
    (UNRELATED, ENTITY, "knows"): [make_source_ref_key(DATASET, OTHER_DOCUMENT)],
}


def _url() -> str:
    username = os.environ.get("DB_USERNAME", "cognee")
    password = os.environ.get("DB_PASSWORD", "cognee")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    database = os.environ.get("DB_NAME", "cognee_db")
    return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"


def _provenance(keys: list[str]) -> dict:
    return {
        "source_ref_keys": keys,
        "source_dataset_ids": [str(DATASET)],
        "source_run_ids": [str(RUN)],
        "source_run_refs": [make_source_run_ref(RUN, key) for key in keys],
    }


_ARRAYS = (
    "CAST(:source_ref_keys AS varchar[]), CAST(:source_dataset_ids AS varchar[]), "
    "CAST(:source_run_ids AS varchar[]), CAST(:source_run_refs AS varchar[])"
)


@pytest_asyncio.fixture
async def graph():
    """An adapter on a schema of its own holding the fork document in the old layout."""
    schema = f"rkf_{uuid4().hex[:16]}"
    admin = create_async_engine(_url())
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    adapter = PostgresDemoAdapter(_url(), schema=schema)
    try:
        async with adapter.engine.begin() as connection:
            for statement in _OLD_LAYOUT:
                await connection.execute(text(statement))
            for node_id, (node_type, properties, keys) in NODES.items():
                await connection.execute(
                    text(
                        "INSERT INTO graph_node (id, name, type, properties, source_ref_keys, "
                        "source_dataset_ids, source_run_ids, source_run_refs) VALUES "
                        f"(:id, :name, :type, CAST(:properties AS jsonb), {_ARRAYS})"
                    ),
                    {
                        "id": node_id,
                        "name": properties.get("name"),
                        "type": node_type,
                        "properties": json.dumps(properties),
                        **_provenance(keys),
                    },
                )
            for (source, target, relationship), keys in EDGES.items():
                endpoints = {"source_node_id": source, "target_node_id": target}
                await connection.execute(
                    text(
                        "INSERT INTO graph_edge (source_id, target_id, relationship_name, "
                        "properties, source_ref_keys, source_dataset_ids, source_run_ids, "
                        "source_run_refs) VALUES (:source, :target, :relationship, "
                        f"CAST(:properties AS jsonb), {_ARRAYS})"
                    ),
                    {
                        "source": source,
                        "target": target,
                        "relationship": relationship,
                        "properties": json.dumps({**endpoints, "edge_text": relationship}),
                        **_provenance(keys),
                    },
                )
        yield adapter
    finally:
        await adapter.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest.fixture(autouse=True)
def _no_relational_store(monkeypatch):
    """The keeper check and the ledger read the relational store; this graph has neither."""

    async def no_keepers(fork_rows, dataset_id):
        return set()

    async def no_ledger(id_map, dataset_id):
        return None

    monkeypatch.setattr(migration, "_keeper_blocked_pairs", no_keepers)
    monkeypatch.setattr(migration, "_update_ledger_references", no_ledger)


async def _snapshot(engine) -> dict:
    """Nodes and edges with properties and provenance, without the timestamps."""
    async with engine.connect() as connection:
        nodes = {
            row.id: (row.type, row.properties, list(row.source_ref_keys), list(row.source_run_refs))
            for row in await connection.execute(
                text(
                    "SELECT id, type, properties, source_ref_keys, source_run_refs FROM graph_node"
                )
            )
        }
        edges = {
            (row.source_id, row.target_id, row.relationship_name): (
                row.properties,
                list(row.source_ref_keys),
                list(row.source_run_refs),
            )
            for row in await connection.execute(
                text(
                    "SELECT source_id, target_id, relationship_name, properties, "
                    "source_ref_keys, source_run_refs FROM graph_edge"
                )
            )
        }
    return {"nodes": nodes, "edges": edges}


async def _run(adapter, fork_rows) -> None:
    context = MigrationContext(graph_engine=adapter, vector_engine=None, dataset_id=DATASET)
    await migration._apply(context, fork_rows)


FORWARD = [(LEGACY, CANONICAL, DATASET)]
BACKWARD = [(CANONICAL, LEGACY, DATASET)]


async def test_the_old_layout_takes_the_sql_path(graph):
    found = await migration._postgres_array_layout(graph)

    assert found is not None and found[0] is graph.engine


async def test_a_graph_outside_postgres_takes_the_adapter_path():
    assert await migration._postgres_array_layout(SimpleNamespace()) is None


async def test_the_fork_document_moves_to_its_canonical_id(graph):
    await _run(graph, FORWARD)
    after = await _snapshot(graph.engine)

    nodes, edges = after["nodes"], after["edges"]
    assert LEGACY not in nodes
    assert nodes[CANONICAL][0] == "TextDocument"
    assert nodes[CHUNK][1]["document_id"] == CANONICAL
    # The edge into the document moved, with the endpoint ids it carries.
    assert (CHUNK, LEGACY, "is_part_of") not in edges
    moved_edge = edges[(CHUNK, CANONICAL, "is_part_of")]
    assert moved_edge[0]["target_node_id"] == CANONICAL
    assert moved_edge[0]["source_node_id"] == CHUNK


async def test_every_key_of_the_document_moves_and_no_other(graph):
    await _run(graph, FORWARD)
    after = await _snapshot(graph.engine)

    every_key = [
        key
        for record in [*after["nodes"].values(), *after["edges"].values()]
        for key in record[-2] + record[-1]
    ]
    assert not [key for key in every_key if LEGACY in key]
    assert after["nodes"][CHUNK][2] == [_document_key(CANONICAL), _chunk_key(CANONICAL)]
    assert after["nodes"][CHUNK][3] == [
        make_source_run_ref(RUN, _document_key(CANONICAL)),
        make_source_run_ref(RUN, _chunk_key(CANONICAL)),
    ]
    # The key it already had is kept once, in its place; the other document's stays.
    assert after["nodes"][ENTITY][2] == [
        make_source_ref_key(DATASET, OTHER_DOCUMENT),
        _document_key(CANONICAL),
    ]
    assert after["nodes"][UNRELATED][2] == [make_source_ref_key(DATASET, OTHER_DOCUMENT)]
    assert after["edges"][(UNRELATED, ENTITY, "knows")][1] == [
        make_source_ref_key(DATASET, OTHER_DOCUMENT)
    ]


async def test_running_again_changes_nothing(graph):
    await _run(graph, FORWARD)
    once = await _snapshot(graph.engine)

    await _run(graph, FORWARD)

    assert await _snapshot(graph.engine) == once


async def test_the_downgrade_puts_the_document_back(graph):
    before = await _snapshot(graph.engine)
    await _run(graph, FORWARD)

    await _run(graph, BACKWARD)
    back = await _snapshot(graph.engine)

    # Exactly as before, except the key the half-moved entity also had: its two
    # copies became one on the way forward, so it comes back without the duplicate.
    expected = {**before, "nodes": dict(before["nodes"])}
    entity_type, entity_properties, _keys, _refs = before["nodes"][ENTITY]
    entity_keys = [make_source_ref_key(DATASET, OTHER_DOCUMENT), _document_key(LEGACY)]
    expected["nodes"][ENTITY] = (
        entity_type,
        entity_properties,
        entity_keys,
        [make_source_run_ref(RUN, key) for key in entity_keys],
    )
    assert back == expected
