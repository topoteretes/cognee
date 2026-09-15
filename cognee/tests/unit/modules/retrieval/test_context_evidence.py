from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever


def test_rag_context_evidence_uses_exact_retrieved_chunks():
    dataset_id = uuid4()
    data_id = uuid4()
    chunk_id = uuid4()
    scored_chunk = SimpleNamespace(
        id=chunk_id,
        score=0.125,
        payload={
            "document_id": str(data_id),
            "document_name": "report.pdf",
            "chunk_index": 3,
            "text": "Revenue grew twelve percent.",
        },
    )

    evidence = CompletionRetriever().get_context_evidence(
        [scored_chunk],
        dataset_id=dataset_id,
    )

    assert len(evidence) == 1
    reference = evidence[0]
    assert reference.kind == "segment"
    assert reference.role == "used_as_context"
    assert reference.artifact_id == str(chunk_id)
    assert reference.dataset_id == str(dataset_id)
    assert reference.source_ref_key == make_source_ref_key(dataset_id, data_id)
    assert reference.data_id == str(data_id)
    assert reference.chunk_id == str(chunk_id)
    assert reference.chunk_index == 3
    assert reference.document_name == "report.pdf"
    assert reference.rank == 0
    assert reference.score == 0.125


def test_graph_context_evidence_uses_exact_nodes_and_edge():
    dataset_id = uuid4()
    source = Node("node-a", {"name": "Alpha"})
    target = Node("node-b", {"name": "Beta"})
    edge = Edge(
        source,
        target,
        attributes={
            "relationship_type": "relates_to",
            "edge_object_id": "edge-1",
        },
    )

    evidence = GraphCompletionRetriever().get_context_evidence(
        [edge],
        dataset_id=dataset_id,
    )

    assert [(reference.kind, reference.artifact_id) for reference in evidence] == [
        ("graph_node", "node-a"),
        ("graph_node", "node-b"),
        ("graph_edge", "edge-1"),
    ]
    assert all(reference.role == "used_as_context" for reference in evidence)
    assert all(reference.dataset_id == str(dataset_id) for reference in evidence)
    assert evidence[0].label == "Alpha"
    assert evidence[1].label == "Beta"
    assert evidence[2].source_node_id == "node-a"
    assert evidence[2].target_node_id == "node-b"
    assert evidence[2].relationship_name == "relates_to"


def test_graph_context_evidence_deduplicates_shared_nodes():
    source = Node("node-a", {"name": "Alpha"})
    middle = Node("node-b", {"name": "Beta"})
    target = Node("node-c", {"name": "Gamma"})
    edges = [
        Edge(
            source,
            middle,
            attributes={"relationship_type": "first", "edge_object_id": "edge-1"},
        ),
        Edge(
            middle,
            target,
            attributes={"relationship_type": "second", "edge_object_id": "edge-2"},
        ),
    ]

    evidence = GraphCompletionRetriever().get_context_evidence(edges)

    assert [reference.artifact_id for reference in evidence if reference.kind == "graph_node"] == [
        "node-a",
        "node-b",
        "node-c",
    ]
    assert [reference.artifact_id for reference in evidence if reference.kind == "graph_edge"] == [
        "edge-1",
        "edge-2",
    ]


# ------------------------------------------------------------------ hybrid


def _hybrid_result(chunks=(), entities=(), facts=(), chunk_summaries=None):
    return {
        "chunks": list(chunks),
        "chunk_summaries": dict(chunk_summaries or {}),
        "entities": list(entities),
        "facts": list(facts),
    }


def _scored_chunk(chunk_id, document_id, text="Some passage.", chunk_index=0, score=0.5):
    return SimpleNamespace(
        id=chunk_id,
        score=score,
        payload={
            "id": str(chunk_id),
            "document_id": str(document_id),
            "document_name": "report.pdf",
            "chunk_index": chunk_index,
            "text": text,
        },
    )


def _bullet(source_id, target_id, relationship, edge_object_id=None, source="Alpha", target="Beta"):
    """An edge bullet as ``hybrid.entities._edge_bullet`` builds it."""
    return {
        "text": f"{source} -- {relationship} -- {target}",
        "source": source,
        "target": target,
        "source_id": source_id,
        "relationship": relationship,
        "target_id": target_id,
        "edge_type_id": "edge-type-1",
        "edge_object_id": edge_object_id,
    }


def _entity(entity_id, name, edges=()):
    return {
        "id": entity_id,
        "name": name,
        "description": None,
        "type": "Person",
        "edges": list(edges),
    }


def test_hybrid_context_evidence_cites_chunks_exactly_like_rag():
    dataset_id = uuid4()
    chunk = _scored_chunk(uuid4(), uuid4(), chunk_index=3, score=0.125)

    hybrid = HybridRetriever().get_context_evidence(_hybrid_result(chunks=[chunk]), dataset_id)
    rag = CompletionRetriever().get_context_evidence([chunk], dataset_id=dataset_id)

    assert [reference.model_dump() for reference in hybrid] == [
        reference.model_dump() for reference in rag
    ]


def test_hybrid_context_evidence_cites_entities_their_edges_and_endpoints():
    dataset_id = uuid4()
    chunk = _scored_chunk(uuid4(), uuid4())
    entity = _entity("node-a", "Alpha", edges=[_bullet("node-a", "node-b", "relates_to", "edge-1")])

    evidence = HybridRetriever().get_context_evidence(
        _hybrid_result(chunks=[chunk], entities=[entity]), dataset_id
    )

    assert [(reference.kind, reference.artifact_id) for reference in evidence] == [
        ("segment", str(chunk.id)),
        ("graph_node", "node-a"),
        ("graph_node", "node-b"),
        ("graph_edge", "edge-1"),
    ]
    assert [reference.rank for reference in evidence] == [0, 1, 2, 3]
    assert all(reference.role == "used_as_context" for reference in evidence)
    assert all(reference.dataset_id == str(dataset_id) for reference in evidence)
    assert evidence[1].label == "Alpha"
    assert evidence[2].label == "Beta"
    assert evidence[3].source_node_id == "node-a"
    assert evidence[3].target_node_id == "node-b"
    assert evidence[3].relationship_name == "relates_to"


def test_hybrid_context_evidence_derives_the_edge_id_the_graph_builder_would():
    entity = _entity("node-a", "Alpha", edges=[_bullet("node-a", "node-b", "relates_to")])

    evidence = HybridRetriever().get_context_evidence(_hybrid_result(entities=[entity]))

    edges = [reference for reference in evidence if reference.kind == "graph_edge"]
    assert [edge.artifact_id for edge in edges] == [
        generate_edge_object_id("node-a", "node-b", "relates_to")
    ]


def test_hybrid_context_evidence_never_invents_an_id_for_an_unidentifiable_bullet():
    """A bullet without an endpoint or relationship is rendered as text only; cite nothing for it."""
    no_relationship = _bullet("node-a", "node-b", "relates_to")
    no_relationship["relationship"] = None
    no_target = _bullet("node-a", None, "relates_to")
    entity = _entity("node-a", "Alpha", edges=[no_relationship, no_target])

    evidence = HybridRetriever().get_context_evidence(_hybrid_result(entities=[entity]))

    assert [(reference.kind, reference.artifact_id) for reference in evidence] == [
        ("graph_node", "node-a")
    ]


def test_hybrid_context_evidence_ignores_facts_and_summaries():
    chunk = _scored_chunk("chunk-1", "doc-1")
    result = _hybrid_result(
        chunks=[chunk],
        entities=[_entity("node-a", "Alpha")],
        facts=[{"id": "edge-type-1", "text": "Alpha works at Acme."}],
        chunk_summaries={"chunk-1": "Short summary"},
    )

    evidence = HybridRetriever().get_context_evidence(result)

    assert [(reference.kind, reference.artifact_id) for reference in evidence] == [
        ("segment", "chunk-1"),
        ("graph_node", "node-a"),
    ]


def test_hybrid_context_evidence_deduplicates_shared_endpoints_and_edges():
    shared = _bullet("node-a", "node-b", "relates_to", "edge-1")
    entities = [
        _entity("node-a", "Alpha", edges=[shared]),
        _entity(
            "node-b",
            "Beta",
            edges=[shared, _bullet("node-b", "node-c", "knows", "edge-2", "Beta", "Gamma")],
        ),
    ]

    evidence = HybridRetriever().get_context_evidence(_hybrid_result(entities=entities))

    assert [reference.artifact_id for reference in evidence if reference.kind == "graph_node"] == [
        "node-a",
        "node-b",
        "node-c",
    ]
    assert [reference.artifact_id for reference in evidence if reference.kind == "graph_edge"] == [
        "edge-1",
        "edge-2",
    ]


def test_hybrid_context_evidence_uses_no_label_when_the_name_is_just_the_id():
    entity = _entity("node-a", "node-a")

    evidence = HybridRetriever().get_context_evidence(_hybrid_result(entities=[entity]))

    assert evidence[0].artifact_id == "node-a"
    assert evidence[0].label is None


def test_hybrid_context_evidence_flattens_batch_results_with_continuing_ranks():
    shared_chunk = _scored_chunk("chunk-1", "doc-1")
    batch = [
        _hybrid_result(chunks=[shared_chunk], entities=[_entity("node-a", "Alpha")]),
        _hybrid_result(chunks=[shared_chunk, _scored_chunk("chunk-2", "doc-1", chunk_index=1)]),
    ]

    evidence = HybridRetriever().get_context_evidence(batch)

    assert [(reference.kind, reference.artifact_id) for reference in evidence] == [
        ("segment", "chunk-1"),
        ("segment", "chunk-2"),
        ("graph_node", "node-a"),
    ]
    assert [reference.rank for reference in evidence] == [0, 1, 2]


@pytest.mark.parametrize("retrieved_objects", [None, "not a result", 42, [], ["not a dict"]])
def test_hybrid_context_evidence_returns_nothing_for_unknown_shapes(retrieved_objects):
    assert HybridRetriever().get_context_evidence(retrieved_objects) == []
