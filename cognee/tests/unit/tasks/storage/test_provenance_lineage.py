"""Unit tests for the provenance lineage layer.

Pure and DB/LLM-free: they exercise the lineage builders (source tier, dataset
tier, and the depth-aware orchestrator) directly on plain DataPoints and
lightweight dataset/data-item stand-ins.
"""

from types import SimpleNamespace
from uuid import uuid4

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.DatasetNode import DatasetNode
from cognee.tasks.storage.provenance_lineage import (
    DERIVED_FROM_RELATIONSHIP,
    IN_DATASET_RELATIONSHIP,
    PROVENANCE_EDGE_FLAG,
    ProvenanceConfig,
    build_dataset_lineage,
    build_provenance_lineage,
    build_source_lineage,
    dataset_lineage_node_id,
)


def _document_node(document_id):
    """A minimal stand-in for the Document graph node (id == data_item.id)."""
    return DataPoint(id=document_id)


def _dataset(dataset_id, name="Fleet Ops"):
    return SimpleNamespace(id=dataset_id, name=name)


def _data_item(document_id):
    return SimpleNamespace(id=document_id)


def _content_node(node_id=None):
    """A non-structural extracted node (type defaults to 'DataPoint')."""
    return DataPoint(id=node_id or uuid4())


def _typed_node(type_name, node_id=None):
    """A node with an explicit graph ``type`` (used to simulate structural nodes)."""
    node = DataPoint(id=node_id or uuid4())
    object.__setattr__(node, "type", type_name)
    return node


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


# ── Source tier: <node> -derived_from-> Document ──


def test_source_lineage_links_every_content_node_to_document():
    document_id = uuid4()
    e1, e2 = _content_node(), _content_node()
    nodes = [_document_node(document_id), e1, e2]

    edges = build_source_lineage(nodes, _data_item(document_id))

    triples = {(source, target, rel) for source, target, rel, _ in edges}
    assert (e1.id, document_id, DERIVED_FROM_RELATIONSHIP) in triples
    assert (e2.id, document_id, DERIVED_FROM_RELATIONSHIP) in triples
    # The Document is not linked to itself, and every edge is flagged.
    assert len(edges) == 2
    assert all(props[PROVENANCE_EDGE_FLAG] is True for *_, props in edges)


def test_source_lineage_skips_structural_nodes():
    document_id = uuid4()
    content = _content_node()
    nodes = [
        _document_node(document_id),
        content,
        _typed_node("NodeSet"),
        _typed_node("DatasetNode"),
    ]

    edges = build_source_lineage(nodes, _data_item(document_id))

    assert {source for source, *_ in edges} == {content.id}


def test_source_lineage_empty_when_document_absent():
    assert build_source_lineage([_content_node()], _data_item(uuid4())) == []


def test_source_lineage_edge_text_is_constant():
    """All derived_from edges share one edge_text so they collapse to one EdgeType."""
    document_id = uuid4()
    nodes = [_document_node(document_id), _content_node(), _content_node()]

    edges = build_source_lineage(nodes, _data_item(document_id))

    assert {props["edge_text"] for *_, props in edges} == {DERIVED_FROM_RELATIONSHIP}


def test_merged_node_accumulates_derived_from_per_document():
    """A node recurring across data items gets one derived_from edge per Document."""
    shared_id = uuid4()
    doc_a, doc_b = uuid4(), uuid4()

    edges_a = build_source_lineage(
        [_document_node(doc_a), _content_node(shared_id)], _data_item(doc_a)
    )
    edges_b = build_source_lineage(
        [_document_node(doc_b), _content_node(shared_id)], _data_item(doc_b)
    )

    triples = {(source, target, rel) for source, target, rel, _ in edges_a + edges_b}
    assert (shared_id, doc_a, DERIVED_FROM_RELATIONSHIP) in triples
    assert (shared_id, doc_b, DERIVED_FROM_RELATIONSHIP) in triples


# ── Orchestrator + depth ──


def test_dataset_depth_includes_both_tiers():
    document_id = uuid4()
    nodes = [_document_node(document_id), _content_node()]
    config = ProvenanceConfig(provenance_lineage=True, provenance_lineage_depth="dataset")

    extra_nodes, extra_edges = build_provenance_lineage(
        nodes, _dataset(uuid4()), _data_item(document_id), config
    )

    rels = {edge[2] for edge in extra_edges}
    assert rels == {DERIVED_FROM_RELATIONSHIP, IN_DATASET_RELATIONSHIP}
    assert any(isinstance(node, DatasetNode) for node in extra_nodes)


def test_document_depth_excludes_dataset_tier():
    document_id = uuid4()
    nodes = [_document_node(document_id), _content_node()]
    config = ProvenanceConfig(provenance_lineage=True, provenance_lineage_depth="document")

    extra_nodes, extra_edges = build_provenance_lineage(
        nodes, _dataset(uuid4()), _data_item(document_id), config
    )

    assert {edge[2] for edge in extra_edges} == {DERIVED_FROM_RELATIONSHIP}
    assert extra_nodes == []


def test_disabled_flag_returns_empty():
    document_id = uuid4()
    nodes = [_document_node(document_id), _content_node()]
    config = ProvenanceConfig(provenance_lineage=False)

    assert build_provenance_lineage(nodes, _dataset(uuid4()), _data_item(document_id), config) == (
        [],
        [],
    )


def test_unknown_depth_falls_back_to_dataset():
    document_id = uuid4()
    nodes = [_document_node(document_id), _content_node()]
    config = ProvenanceConfig(provenance_lineage=True, provenance_lineage_depth="bogus")

    _, extra_edges = build_provenance_lineage(
        nodes, _dataset(uuid4()), _data_item(document_id), config
    )

    # Fell back to the default (dataset) depth, so the dataset tier is present.
    assert IN_DATASET_RELATIONSHIP in {edge[2] for edge in extra_edges}
