"""Unit tests for the Dataset tier of the provenance lineage layer.

Pure and DB/LLM-free: they exercise ``build_dataset_lineage`` directly on plain
DataPoints and lightweight dataset/data-item stand-ins.
"""

from types import SimpleNamespace
from uuid import uuid4

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.DatasetNode import DatasetNode
from cognee.tasks.storage.provenance_lineage import (
    IN_DATASET_RELATIONSHIP,
    PROVENANCE_EDGE_FLAG,
    build_dataset_lineage,
    dataset_lineage_node_id,
)


def _document_node(document_id):
    """A minimal stand-in for the Document graph node (id == data_item.id)."""
    return DataPoint(id=document_id)


def _dataset(dataset_id, name="Fleet Ops"):
    return SimpleNamespace(id=dataset_id, name=name)


def _data_item(document_id):
    return SimpleNamespace(id=document_id)


def test_emits_dataset_node_and_in_dataset_edge():
    document_id = uuid4()
    dataset_id = uuid4()
    nodes = [_document_node(document_id)]

    extra_nodes, extra_edges = build_dataset_lineage(
        nodes, _dataset(dataset_id), _data_item(document_id)
    )

    assert len(extra_nodes) == 1
    assert isinstance(extra_nodes[0], DatasetNode)
    assert extra_nodes[0].name == "Fleet Ops"

    assert len(extra_edges) == 1
    source, target, relationship, properties = extra_edges[0]
    assert source == document_id
    assert target == dataset_lineage_node_id(dataset_id)
    assert relationship == IN_DATASET_RELATIONSHIP
    assert properties[PROVENANCE_EDGE_FLAG] is True


def test_dataset_node_id_is_deterministic_and_shared():
    """Same dataset id -> same lineage node id, so the node dedups across data items."""
    dataset_id = uuid4()
    doc_a, doc_b = uuid4(), uuid4()

    nodes_a, _ = build_dataset_lineage(
        [_document_node(doc_a)], _dataset(dataset_id), _data_item(doc_a)
    )
    nodes_b, _ = build_dataset_lineage(
        [_document_node(doc_b)], _dataset(dataset_id), _data_item(doc_b)
    )

    assert nodes_a[0].id == nodes_b[0].id == dataset_lineage_node_id(dataset_id)


def test_different_datasets_get_distinct_nodes():
    doc = uuid4()
    nodes_1, _ = build_dataset_lineage([_document_node(doc)], _dataset(uuid4()), _data_item(doc))
    nodes_2, _ = build_dataset_lineage([_document_node(doc)], _dataset(uuid4()), _data_item(doc))
    assert nodes_1[0].id != nodes_2[0].id


def test_no_edge_when_document_node_absent():
    """Without the Document node in the batch, no dangling edge is produced."""
    extra_nodes, extra_edges = build_dataset_lineage(
        [_document_node(uuid4())], _dataset(uuid4()), _data_item(uuid4())
    )
    assert extra_nodes == []
    assert extra_edges == []


def test_falls_back_to_dataset_id_when_name_missing():
    document_id = uuid4()
    dataset_id = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name=None)

    extra_nodes, _ = build_dataset_lineage(
        [_document_node(document_id)], dataset, _data_item(document_id)
    )
    assert extra_nodes[0].name == str(dataset_id)


def test_returns_empty_when_dataset_or_data_item_missing():
    document_id = uuid4()
    nodes = [_document_node(document_id)]
    assert build_dataset_lineage(nodes, None, _data_item(document_id)) == ([], [])
    assert build_dataset_lineage(nodes, _dataset(uuid4()), None) == ([], [])
