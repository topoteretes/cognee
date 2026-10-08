"""The structure planner and the cognify hook around it, without a graph (SDK-986).

The reconcile against real adapters is in
``tests/integration/infrastructure/graph/test_document_structure_contract.py``.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.engine.models import StructureContainer
from cognee.tasks.ingestion import document_structure
from cognee.tasks.ingestion.document_structure import StructureRow, plan_document_structure

DATASET_ID = uuid4()


def _row(external_id, ancestors, rank=0.0):
    return StructureRow(
        data_id=uuid4(),
        document_type="TextDocument",
        source="notion",
        table_name="pages",
        external_id=external_id,
        ancestors=tuple(ancestors),
        rank=rank,
    )


def _page(external_id):
    return {"kind": "page", "id": external_id, "document": True}


def _container(kind, external_id):
    return str(StructureContainer.container_id(DATASET_ID, "notion", "pages", kind, external_id))


def test_a_row_hangs_under_its_parent_row_and_its_containers_chain_up_to_a_page():
    page = _row("p", [])
    sub = _row("s", [_page("p")])
    chain = [{"kind": "data_source", "id": "ds"}, {"kind": "database", "id": "db"}, _page("p")]
    row = _row("r", chain)

    plan = plan_document_structure([page, sub, row], DATASET_ID)

    assert plan.edges == {
        (str(sub.data_id), str(page.data_id)),
        (str(row.data_id), _container("data_source", "ds")),
        (_container("data_source", "ds"), _container("database", "db")),
        (_container("database", "db"), str(page.data_id)),
    }


def test_a_parent_that_is_not_in_the_dataset_gets_no_edge():
    row = _row("c", [_page("outside the selected roots")])

    plan = plan_document_structure([row], DATASET_ID)

    assert plan.edges == set()
    assert plan.unresolved == 1


def test_the_newest_of_two_rows_with_one_external_id_is_the_parent():
    """add(run_in_background=True) skips orphan cleanup, so an edited page can keep its
    old row next to the new one until the next sync."""
    stale, fresh = _row("p", [], rank=1.0), _row("p", [], rank=2.0)
    child = _row("c", [_page("p")])

    plan = plan_document_structure([fresh, stale, child], DATASET_ID)

    assert plan.edges == {(str(child.data_id), str(fresh.data_id))}


@pytest.mark.asyncio
async def test_a_dataset_without_structure_does_not_touch_the_graph(monkeypatch):
    """The pass runs on every cognify of every dataset."""

    async def no_graph():
        raise AssertionError("the graph was opened")

    async def rows(_dataset_id):
        drive_row = {"source": "google_drive", "table_name": "t", "external_id": "1"}
        return [SimpleNamespace(id=uuid4(), system_metadata=drive_row, extension="txt")]

    monkeypatch.setattr(document_structure, "get_graph_engine", no_graph)
    monkeypatch.setattr(document_structure, "dataset_rows", rows)
    monkeypatch.setattr(
        document_structure, "current_dataset_id", SimpleNamespace(get=lambda: DATASET_ID)
    )

    assert await document_structure.reconcile_document_structure() is None


def test_two_rows_that_name_a_container_differently_leave_it_the_newest_name():
    old = _row("a", [{"kind": "database", "id": "db", "name": "Old"}], rank=1.0)
    new = _row("b", [{"kind": "database", "id": "db", "name": "New"}], rank=2.0)

    plan = plan_document_structure([new, old], DATASET_ID)

    assert [container.name for container in plan.containers.values()] == ["New"]


def test_child_of_counts_as_structure_for_visualization_and_contradictions():
    from cognee.modules.engine.models import CHILD_OF
    from cognee.modules.visualization.preprocessor import _STRUCTURAL_RELATIONS
    from cognee.tasks.graph.detect_contradictions import STRUCTURAL_RELATIONSHIPS

    assert CHILD_OF in _STRUCTURAL_RELATIONS
    assert CHILD_OF in STRUCTURAL_RELATIONSHIPS
