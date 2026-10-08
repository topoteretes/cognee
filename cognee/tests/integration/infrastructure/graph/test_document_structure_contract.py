"""The document structure pass against real graph adapters (SDK-986).

Runs on every backend ``graph_provenance_adapter`` covers (Ladybug always, Postgres
and Neo4j when ``.env`` configures them). No LLM and no source: rows are described
directly and the Document nodes they stand for are written the way cognify would.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
    make_source_ref_key,
)
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import CHILD_OF, StructureContainer
from cognee.tasks.ingestion.document_structure import (
    StructureRow,
    plan_document_structure,
    reconcile_structure,
)

pytestmark = pytest.mark.asyncio

DATASET_ID = uuid4()
DOCUMENT_TYPES = {"TextDocument"}


def _page(external_id):
    return {"kind": "page", "id": external_id, "document": True}


def _data_source(external_id="ds", name="Tasks"):
    return {"kind": "data_source", "id": external_id, "name": name}


def _database(external_id="db", name="Tasks db"):
    return {"kind": "database", "id": external_id, "name": name}


class _Tree:
    """The rows of one Notion-shaped dataset, keyed by external id."""

    def __init__(self):
        self.rows: dict[str, StructureRow] = {}

    def row(self, external_id, ancestors, rank=0.0, data_id=None):
        self.rows[external_id] = StructureRow(
            data_id=data_id or uuid4(),
            document_type="TextDocument",
            source="notion",
            table_name="notion_pages",
            external_id=external_id,
            ancestors=tuple(ancestors),
            rank=rank,
        )
        return self.rows[external_id].data_id

    def plan(self, dataset_id=DATASET_ID):
        return plan_document_structure(list(self.rows.values()), dataset_id)

    def documents(self):
        return [
            TextDocument(
                id=row.data_id,
                name=row.external_id,
                raw_data_location="x",
                external_metadata=None,
                mime_type="text/plain",
            )
            for row in self.rows.values()
        ]


class _Recorder:
    """The adapter, recording every write the pass makes through it."""

    WRITES = (
        "add_nodes",
        "add_edges",
        "delete_nodes",
        "delete_edge_triples",
        "attach_node_source_refs",
        "remove_node_source_refs",
    )

    def __init__(self, adapter):
        self._adapter, self.writes = adapter, []

    def __getattr__(self, name):
        attribute = getattr(self._adapter, name)
        if name not in self.WRITES:
            return attribute

        async def recorded(*args, **kwargs):
            self.writes.append(name)
            return await attribute(*args, **kwargs)

        return recorded


async def _child_of(adapter) -> set[tuple[str, str]]:
    _nodes, edges = await adapter.get_graph_data()
    return {(str(edge[0]), str(edge[1])) for edge in edges if edge[2] == CHILD_OF}


async def _container_ids(adapter) -> set[str]:
    nodes, _edges = await adapter.get_graph_data()
    return {
        str(node_id)
        for node_id, properties in nodes
        if properties.get("type") == StructureContainer.__name__
    }


async def _reconcile(adapter, tree, dataset_id=DATASET_ID):
    return await reconcile_structure(adapter, dataset_id, tree.plan(dataset_id), DOCUMENT_TYPES)


def _container(kind, external_id, dataset_id=DATASET_ID):
    return str(
        StructureContainer.container_id(dataset_id, "notion", "notion_pages", kind, external_id)
    )


def _tree():
    """root page, a sub-page, and a database with one row, all under the root."""
    tree = _Tree()
    tree.row("root", [])
    tree.row("sub", [_page("root")])
    tree.row("row", [_data_source(), _database(), _page("root")])
    return tree


async def _mark_provenance(adapter):
    """Make the graph store its provenance in the graph, as one that was empty when
    cognee first wrote to it does."""
    await adapter.set_graph_metadata(
        {
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        }
    )


async def _synced(adapter, tree):
    await adapter.add_nodes(tree.documents())
    return await _reconcile(adapter, tree)


async def test_every_row_hangs_under_its_real_parent(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()

    stats = await _synced(adapter, tree)

    ids = {key: str(row.data_id) for key, row in tree.rows.items()}
    data_source, database = _container("data_source", "ds"), _container("database", "db")
    assert await _child_of(adapter) == {
        (ids["sub"], ids["root"]),
        (ids["row"], data_source),
        (data_source, database),
        (database, ids["root"]),
    }
    assert await _container_ids(adapter) == {data_source, database}
    assert stats["edges_added"] == 4 and stats["containers_added"] == 2


@pytest.mark.parametrize("provenance", [False, True], ids=["ledger", "provenance"])
async def test_a_second_run_with_nothing_changed_writes_nothing(
    graph_provenance_adapter, provenance
):
    adapter, tree = graph_provenance_adapter, _tree()
    if provenance:
        # The stamping of containers with their rows' refs is a write of its own.
        await _mark_provenance(adapter)
    await _synced(adapter, tree)

    recorder = _Recorder(adapter)
    stats = await _reconcile(recorder, tree)

    # The writes themselves, not the numbers the pass reports about them.
    assert recorder.writes == []
    assert stats["edges_added"] == stats["edges_removed"] == 0
    assert stats["containers_added"] == stats["containers_removed"] == 0


async def test_a_moved_row_keeps_exactly_one_parent_edge(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    other = tree.row("other", [])
    await _synced(adapter, tree)

    # The sub-page moves under another page, without its content (and id) changing.
    tree.row("sub", [_page("other")], data_id=tree.rows["sub"].data_id)
    stats = await _reconcile(adapter, tree)

    sub = str(tree.rows["sub"].data_id)
    assert {edge for edge in await _child_of(adapter) if edge[0] == sub} == {(sub, str(other))}
    assert stats["edges_added"] == 1 and stats["edges_removed"] == 1


async def test_a_row_moved_to_another_database_removes_the_container_it_left(
    graph_provenance_adapter,
):
    adapter, tree = graph_provenance_adapter, _tree()
    await _synced(adapter, tree)

    tree.row(
        "row",
        [_data_source("ds2", "Other"), _database("db2", "Other db"), _page("root")],
        data_id=tree.rows["row"].data_id,
    )
    stats = await _reconcile(adapter, tree)

    assert await _container_ids(adapter) == {
        _container("data_source", "ds2"),
        _container("database", "db2"),
    }
    row = str(tree.rows["row"].data_id)
    assert {edge for edge in await _child_of(adapter) if edge[0] == row} == {
        (row, _container("data_source", "ds2"))
    }
    assert stats["containers_removed"] == 2


async def test_an_edge_waits_for_the_document_it_points_at(graph_provenance_adapter):
    """An empty page has no Document node, and a failed cognify writes none. The edge
    is neither written nor reported, and appears once both ends exist."""
    adapter, tree = graph_provenance_adapter, _Tree()
    root = tree.row("root", [])
    child = tree.row("child", [_page("root")])
    await adapter.add_nodes([doc for doc in tree.documents() if doc.id == root])

    first = await _reconcile(adapter, tree)
    assert await _child_of(adapter) == set()
    assert first["edges_added"] == 0 and first["edges_waiting_for_a_document"] == 1

    await adapter.add_nodes([doc for doc in tree.documents() if doc.id == child])
    second = await _reconcile(adapter, tree)
    assert await _child_of(adapter) == {(str(child), str(root))}
    assert second["edges_added"] == 1

    recorder = _Recorder(adapter)
    await _reconcile(recorder, tree)
    assert recorder.writes == []


async def test_a_row_that_moved_out_of_a_container_gives_up_its_claim_on_it(
    graph_provenance_adapter,
):
    adapter, tree = graph_provenance_adapter, _tree()
    await _mark_provenance(adapter)
    sibling = tree.row("row2", [_data_source(), _database(), _page("root")])
    await _synced(adapter, tree)
    mover = tree.rows["row"].data_id

    tree.row("row", [_page("root")], data_id=mover)
    await _reconcile(adapter, tree)

    database = _container("database", "db")
    snapshot = (await adapter.get_node_delete_data([database]))[database]
    # Only the sibling still holds the database: forgetting it removes the container.
    assert set(snapshot.source_ref_keys) == {make_source_ref_key(DATASET_ID, sibling)}


async def test_a_renamed_container_is_written_and_what_is_not_ours_is_left_alone(
    graph_provenance_adapter,
):
    """The pass removes only this dataset's containers and only the edges of nodes it
    manages: a stranger's container and a stranger's child_of edge stay."""
    adapter, tree = graph_provenance_adapter, _tree()
    other_dataset = uuid4()
    stranger = StructureContainer(
        id=StructureContainer.container_id(
            other_dataset, "notion", "notion_pages", "database", "x"
        ),
        name="x",
        kind="database",
        external_id="x",
        source="notion",
        table_name="notion_pages",
        dataset_id=str(other_dataset),
    )
    loose_child, loose_parent = _Tree(), _Tree()
    child, parent = loose_child.row("a", []), loose_parent.row("b", [])
    await adapter.add_nodes([stranger, *loose_child.documents(), *loose_parent.documents()])
    await adapter.add_edges([(str(child), str(parent), CHILD_OF, {"relationship_name": CHILD_OF})])
    await _synced(adapter, tree)

    tree.row(
        "row",
        [_data_source(), _database(name="Renamed"), _page("root")],
        data_id=tree.rows["row"].data_id,
    )
    stats = await _reconcile(adapter, tree)

    assert (await adapter.get_node(_container("database", "db")))["name"] == "Renamed"
    assert stats["containers_renamed"] == 1 and stats["containers_removed"] == 0
    assert str(stranger.id) in await _container_ids(adapter)
    assert (str(child), str(parent)) in await _child_of(adapter)
