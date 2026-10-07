"""The structure planner and the cognify hook around it (SDK-986), without a graph.

The reconcile against real adapters is in
``tests/integration/infrastructure/graph/test_document_structure_contract.py``.
"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.modules.engine.models import CHILD_OF, StructureContainer
from cognee.tasks.ingestion import document_structure
from cognee.tasks.ingestion.document_structure import (
    StructureRow,
    plan_document_structure,
    reconcile_structure,
    structure_rows,
)

DATASET_ID = uuid4()


def _row(external_id, ancestors, rank=0.0, source="notion", table="pages", data_id=None):
    return StructureRow(
        data_id=data_id or uuid4(),
        document_type="TextDocument",
        source=source,
        table_name=table,
        external_id=external_id,
        ancestors=tuple(ancestors),
        rank=rank,
    )


def _page(external_id):
    return {"kind": "page", "id": external_id, "document": True}


def _database(external_id="db", name="Tasks"):
    return {"kind": "database", "id": external_id, "name": name}


def test_a_row_hangs_under_the_row_its_parent_names_not_under_a_data_id():
    parent, child = _row("p", []), _row("c", [_page("p")])

    plan = plan_document_structure([parent, child], DATASET_ID)

    assert plan.edges == {(str(child.data_id), str(parent.data_id))}
    assert plan.containers == {}


def test_a_chain_of_containers_links_each_to_the_next_and_ends_at_the_page():
    page, row = (
        _row("p", []),
        _row("r", [{"kind": "data_source", "id": "ds"}, _database(), _page("p")]),
    )

    plan = plan_document_structure([page, row], DATASET_ID)

    data_source = str(
        StructureContainer.container_id(DATASET_ID, "notion", "pages", "data_source", "ds")
    )
    database = str(StructureContainer.container_id(DATASET_ID, "notion", "pages", "database", "db"))
    assert plan.edges == {
        (str(row.data_id), data_source),
        (data_source, database),
        (database, str(page.data_id)),
    }
    assert plan.owners == {
        StructureContainer.container_id(DATASET_ID, "notion", "pages", "data_source", "ds"): {
            row.data_id
        },
        StructureContainer.container_id(DATASET_ID, "notion", "pages", "database", "db"): {
            row.data_id
        },
    }


def test_a_parent_that_is_not_in_the_dataset_gets_no_edge_and_is_counted():
    row = _row("c", [_database(), _page("missing")])

    plan = plan_document_structure([row], DATASET_ID)

    database = str(StructureContainer.container_id(DATASET_ID, "notion", "pages", "database", "db"))
    assert plan.edges == {(str(row.data_id), database)}
    assert plan.unresolved == 1


def test_a_row_is_never_matched_across_sources_or_tables():
    other_table = _row("p", [], table="other")
    other_source = _row("p", [], source="drive")
    row = _row("c", [_page("p")])

    plan = plan_document_structure([other_table, other_source, row], DATASET_ID)

    assert plan.edges == set()
    assert plan.unresolved == 1


def test_two_rows_with_one_external_id_resolve_to_the_newest():
    stale, fresh = _row("p", [], rank=1.0), _row("p", [], rank=2.0)
    child = _row("c", [_page("p")])

    plan = plan_document_structure([stale, fresh, child], DATASET_ID)

    assert plan.edges == {(str(child.data_id), str(fresh.data_id))}


def test_the_newest_row_names_a_container():
    old = _row("a", [_database(name="Old")], rank=1.0)
    new = _row("b", [_database(name="New")], rank=2.0)

    plan = plan_document_structure([new, old], DATASET_ID)

    assert [container.name for container in plan.containers.values()] == ["New"]
    assert set(next(iter(plan.owners.values()))) == {old.data_id, new.data_id}


def test_a_container_without_a_name_is_named_after_its_id():
    plan = plan_document_structure([_row("a", [{"kind": "folder", "id": "f1"}])], DATASET_ID)

    assert [container.name for container in plan.containers.values()] == ["f1"]


def test_two_datasets_never_share_a_container():
    rows = [_row("a", [_database()])]

    mine = plan_document_structure(rows, DATASET_ID)
    theirs = plan_document_structure(rows, uuid4())

    assert set(mine.containers).isdisjoint(theirs.containers)


class _Data:
    def __init__(self, metadata, extension="txt"):
        self.id = uuid4()
        self.system_metadata = metadata
        self.extension = extension
        self.updated_at = None
        self.created_at = None


def _meta(**extra):
    return {
        "source": "notion",
        "table_name": "pages",
        "external_id": "x",
        "structure": {"ancestors": []},
        **extra,
    }


def test_only_rows_that_carry_a_structure_are_read():
    carrying = _Data(_meta())
    without = _Data({"source": "notion", "table_name": "pages", "external_id": "y"})
    other_kind = _Data(None)
    no_scope = _Data({"source": "notion", "structure": {"ancestors": []}})

    rows = structure_rows([carrying, without, other_kind, no_scope])

    assert [row.data_id for row in rows] == [carrying.id]
    assert rows[0].document_type == "TextDocument"


def _in_dataset(monkeypatch):
    monkeypatch.setattr(
        document_structure, "current_dataset_id", SimpleNamespace(get=lambda: DATASET_ID)
    )


class _Graph:
    """Just enough graph for the hook: records writes, optionally refuses edge deletion."""

    def __init__(self, nodes=(), edges=(), can_delete_edges=True):
        self.nodes, self.edges = list(nodes), list(edges)
        self.can_delete_edges = can_delete_edges
        self.added_edges, self.added_nodes, self.deleted_edges = [], [], []

    async def get_graph_metadata(self):
        raise UnsupportedProvenanceCapability()

    async def get_filtered_graph_data(self, _filters):
        return self.nodes, self.edges

    async def add_nodes(self, nodes, **_stamp):
        self.added_nodes.extend(nodes)

    async def add_edges(self, edges, **_stamp):
        self.added_edges.extend(edges)

    async def delete_edge_triples(self, edges):
        if not self.can_delete_edges:
            raise UnsupportedProvenanceCapability()
        self.deleted_edges.extend(edges)


@pytest.mark.asyncio
async def test_a_backend_that_cannot_delete_edges_keeps_the_stale_one_and_still_adds():
    parent, other, child = _row("p", []), _row("o", []), _row("c", [_page("o")])
    stale = (str(child.data_id), str(parent.data_id), CHILD_OF, {})
    graph = _Graph(edges=[stale], can_delete_edges=False)
    plan = plan_document_structure([parent, other, child], DATASET_ID)

    stats = await reconcile_structure(graph, DATASET_ID, plan, {"TextDocument"})

    assert [edge[:3] for edge in graph.added_edges] == [
        (str(child.data_id), str(other.data_id), CHILD_OF)
    ]
    assert stats["edges_added"] == 1 and stats["edges_removed"] == 0


@pytest.mark.asyncio
async def test_a_dataset_without_structure_does_not_touch_the_graph(monkeypatch):
    async def no_graph():
        raise AssertionError("the graph was opened")

    async def rows(_dataset_id):
        return [_Data({"source": "drive", "table_name": "t", "external_id": "1"})]

    monkeypatch.setattr(document_structure, "get_graph_engine", no_graph)
    monkeypatch.setattr(document_structure, "get_dataset_data", rows)
    _in_dataset(monkeypatch)

    assert await document_structure.reconcile_document_structure() is None


@pytest.mark.asyncio
async def test_a_failing_pass_is_swallowed_so_it_cannot_fail_cognify(monkeypatch):
    async def broken(_dataset_id):
        raise RuntimeError("relational database is down")

    monkeypatch.setattr(document_structure, "get_dataset_data", broken)
    _in_dataset(monkeypatch)

    assert await document_structure.reconcile_document_structure() is None


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed(monkeypatch):
    async def cancelled(_dataset_id):
        raise asyncio.CancelledError

    monkeypatch.setattr(document_structure, "get_dataset_data", cancelled)
    _in_dataset(monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        await document_structure.reconcile_document_structure()


def test_child_of_counts_as_structure_for_visualization_and_contradiction_detection():
    from cognee.modules.visualization.preprocessor import _STRUCTURAL_RELATIONS
    from cognee.tasks.graph.detect_contradictions import STRUCTURAL_RELATIONSHIPS

    assert CHILD_OF in _STRUCTURAL_RELATIONS
    assert CHILD_OF in STRUCTURAL_RELATIONSHIPS
