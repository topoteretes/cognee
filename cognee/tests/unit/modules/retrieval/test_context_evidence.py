from types import SimpleNamespace
from uuid import uuid4

from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


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
