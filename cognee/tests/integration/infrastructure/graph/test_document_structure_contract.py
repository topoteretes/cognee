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


async def test_a_second_run_with_nothing_changed_writes_nothing(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    await _synced(adapter, tree)

    stats = await _reconcile(adapter, tree)

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


async def test_a_container_that_is_renamed_is_updated_in_place(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    await _synced(adapter, tree)

    tree.row(
        "row",
        [_data_source(), _database(name="Renamed"), _page("root")],
        data_id=tree.rows["row"].data_id,
    )
    stats = await _reconcile(adapter, tree)

    node = await adapter.get_node(_container("database", "db"))
    assert node["name"] == "Renamed"
    assert stats["containers_added"] == 1 and stats["containers_removed"] == 0
    assert stats["edges_added"] == stats["edges_removed"] == 0


async def test_editing_a_parent_keeps_the_edges_of_its_unchanged_children(
    graph_provenance_adapter,
):
    """An edit gives the parent a new Document id; cognify deletes the old node with
    every edge the children had to it, and does not reprocess the children."""
    adapter, tree = graph_provenance_adapter, _tree()
    await _synced(adapter, tree)

    old_root = tree.rows["root"].data_id
    await adapter.delete_node(str(old_root))
    new_root = tree.row("root", [])
    await adapter.add_nodes([doc for doc in tree.documents() if doc.id == new_root])
    await _reconcile(adapter, tree)

    sub, database = str(tree.rows["sub"].data_id), _container("database", "db")
    assert {(sub, str(new_root)), (database, str(new_root))} <= await _child_of(adapter)
    assert not [edge for edge in await _child_of(adapter) if str(old_root) in edge]


async def test_a_parent_outside_the_dataset_gets_no_edge(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _Tree()
    tree.row("orphan", [_page("not-synced")])
    await adapter.add_nodes(tree.documents())

    stats = await _reconcile(adapter, tree)

    assert await _child_of(adapter) == set()
    assert stats["unresolved_parents"] == 1


async def test_deleting_a_parent_leaves_its_children_without_an_edge(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    await _synced(adapter, tree)

    await adapter.delete_node(str(tree.rows["root"].data_id))
    del tree.rows["root"]
    await _reconcile(adapter, tree)

    sub = str(tree.rows["sub"].data_id)
    assert await adapter.has_node(sub) is True
    assert not [edge for edge in await _child_of(adapter) if sub in edge]


async def test_rows_without_structure_keep_their_own_child_of_edges(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    stranger_child, stranger_parent = uuid4(), uuid4()
    other = _Tree()
    other.row("a", [], data_id=stranger_child)
    other.row("b", [], data_id=stranger_parent)
    await adapter.add_nodes(other.documents())
    await adapter.add_edges(
        [(str(stranger_child), str(stranger_parent), CHILD_OF, {"relationship_name": CHILD_OF})]
    )

    await _synced(adapter, tree)

    assert (str(stranger_child), str(stranger_parent)) in await _child_of(adapter)


async def test_containers_are_owned_by_every_row_beneath_them(graph_provenance_adapter):
    adapter, tree = graph_provenance_adapter, _tree()
    await adapter.set_graph_metadata(
        {
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        }
    )
    second = tree.row("row2", [_data_source(), _database(), _page("root")])
    await _synced(adapter, tree)

    snapshot = await adapter.get_node_delete_data(
        [_container("data_source", "ds"), _container("database", "db")]
    )
    owners = {
        make_source_ref_key(DATASET_ID, tree.rows["row"].data_id),
        make_source_ref_key(DATASET_ID, second),
    }
    for node_id in (_container("data_source", "ds"), _container("database", "db")):
        assert set(snapshot[node_id].source_ref_keys) == owners

    # Forgetting one row leaves the container to the other, as the planner decides by refs.
    await adapter.remove_node_source_refs(
        [_container("database", "db")], [make_source_ref_key(DATASET_ID, second)]
    )
    assert await adapter.has_node(_container("database", "db")) is True


async def test_two_datasets_syncing_one_workspace_share_no_container():
    first, second = _tree(), _tree()
    other_dataset = uuid4()

    mine, theirs = first.plan(DATASET_ID), second.plan(other_dataset)

    assert set(mine.containers).isdisjoint(theirs.containers)
