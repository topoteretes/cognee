"""Re-adding an existing node or edge keeps what improve() learned about it.

``feedback_weight`` (and the truth-subspace state on nodes) is written by
improve() after ingestion; a DataPoint only carries the default. A later
cognify that extracts the same entity, a persisted session window or an update
re-merges the element and must not reset those values.
"""

from unittest.mock import AsyncMock
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties


class Thing(DataPoint):
    name: str
    description: str = ""
    metadata: dict = {"index_fields": ["name"]}


def _thing(name: str, description: str = "") -> Thing:
    return Thing(id=uuid5(NAMESPACE_OID, name), name=name, description=description)


@pytest.fixture
def ladybug(tmp_path):
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    return LadybugAdapter(db_path=str(tmp_path / "learned.lbug"))


@pytest.mark.asyncio
async def test_ladybug_node_keeps_feedback_weight_and_truth_state_across_add_nodes(ladybug):
    first = _thing("TechCorp", "a company")
    node_id = str(first.id)
    await ladybug.add_nodes([first])
    await ladybug.set_node_feedback_weights({node_id: 0.9})
    await ladybug.set_node_truth_state({node_id: {"truth_alignment": [1.0, 0.0], "truth_epoch": 3}})

    await ladybug.add_nodes([_thing("TechCorp", "a company that focuses on AI")])

    assert (await ladybug.get_node_feedback_weights([node_id]))[node_id] == 0.9
    assert (await ladybug.get_node_truth_state([node_id]))[node_id] == {
        "truth_alignment": [1.0, 0.0],
        "truth_epoch": 3,
    }
    node = await ladybug.get_node(node_id)
    assert node["description"] == "a company that focuses on AI"


@pytest.mark.asyncio
async def test_ladybug_new_node_gets_the_default_weight(ladybug):
    fresh = _thing("Fresh")
    await ladybug.add_nodes([fresh])
    assert (await ladybug.get_node_feedback_weights([str(fresh.id)]))[str(fresh.id)] == 0.5


@pytest.mark.asyncio
async def test_ladybug_edge_keeps_feedback_weight_across_add_edges_and_add_edge(ladybug):
    source, target = _thing("Cognee"), _thing("TechCorp")
    await ladybug.add_nodes([source, target])
    src, tgt = str(source.id), str(target.id)
    edges = ensure_default_edge_properties([(src, tgt, "helps", {"edge_text": "first"})])
    edge_object_id = edges[0][3]["edge_object_id"]
    await ladybug.add_edges(edges)
    await ladybug.set_edge_feedback_weights({edge_object_id: 0.8})

    await ladybug.add_edges(
        ensure_default_edge_properties([(src, tgt, "helps", {"edge_text": "second"})])
    )
    assert (await ladybug.get_edge_feedback_weights([edge_object_id]))[edge_object_id] == 0.8

    await ladybug.add_edge(
        src, tgt, "helps", ensure_default_edge_properties([(src, tgt, "helps", {})])[0][3]
    )
    assert (await ladybug.get_edge_feedback_weights([edge_object_id]))[edge_object_id] == 0.8


def _neo4j():
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    adapter = object.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(return_value=[])
    return adapter


def _statements(adapter) -> list[str]:
    return [call.args[0] for call in adapter.query.await_args_list]


def _keeps_feedback_weight(statement: str, alias: str) -> bool:
    kept = f"{alias}.feedback_weight AS kept_feedback_weight" in statement
    restored = (
        f"SET {alias}.feedback_weight = coalesce(kept_feedback_weight, {alias}.feedback_weight)"
        in statement
    )
    # The stored value is captured before the properties are merged in, and
    # written back in its own SET so it reads the merged value for a new element.
    return kept and restored and statement.index("kept_feedback_weight") < statement.index("+=")


@pytest.mark.asyncio
async def test_neo4j_node_merges_keep_the_stored_feedback_weight():
    adapter = _neo4j()
    await adapter.add_node(_thing("TechCorp"))
    await adapter.add_nodes([_thing("TechCorp")])
    single, batch = _statements(adapter)
    assert _keeps_feedback_weight(single, "node")
    assert _keeps_feedback_weight(batch, "n")


@pytest.mark.asyncio
async def test_neo4j_edge_merges_keep_the_stored_feedback_weight():
    adapter = _neo4j()
    await adapter.add_edge("a", "b", "helps", {"feedback_weight": 0.5})
    await adapter.add_edges([("a", "b", "helps", {"feedback_weight": 0.5})])
    single, batch = _statements(adapter)
    assert _keeps_feedback_weight(single, "r")
    assert _keeps_feedback_weight(batch, "rel")
