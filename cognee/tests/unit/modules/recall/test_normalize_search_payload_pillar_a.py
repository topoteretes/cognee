"""End-to-end wiring tests for Pillar A on ``normalize_search_payload``.

Every ``SearchResultItem`` returned by ``normalize_search_payload``
must now carry ``relevance`` (when a score is available) and
``citations`` (when the payload kind is a graph completion). These
tests pin the wiring against the shape the graph-completion retriever
actually produces so a regression in the normalizer surfaces before it
reaches an agent.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

from cognee.modules.recall.methods.normalize_search_payload import (
    normalize_search_payload,
)
from cognee.modules.recall.types.SearchResultItem import (
    SearchResultItem,
    SearchResultKind,
)
from cognee.modules.retrieval.utils.citation_models import CitationKind
from cognee.modules.retrieval.utils.confidence import Confidence, derive_confidence
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def _edge(source: str, target: str, edge_id: str) -> SimpleNamespace:
    """Return a minimal Edge-shaped stand-in for the graph path.

    Matches the duck-type ``used_graph_elements.is_edge_list`` checks for
    on real ``CogneeGraphElements.Edge`` instances: ``node1`` and ``node2``
    with ``.id``, and ``attributes["edge_object_id"]`` for the edge id.
    """
    return SimpleNamespace(
        node1=SimpleNamespace(id=source),
        node2=SimpleNamespace(id=target),
        attributes={"edge_object_id": edge_id},
    )


def _graph_completion_payload(
    completions: list[str] | None = None,
    triplets: list[SimpleNamespace] | None = None,
    dataset_id: UUID | None = None,
    dataset_name: str = "test-dataset",
) -> SearchResultPayload:
    """Build a payload shaped like what ``get_retriever_output`` emits."""
    return SearchResultPayload(
        result_object=triplets or [],
        context="node-a is connected to node-b via rel",
        completion=completions or ["some grounded answer"],
        search_type=SearchType.GRAPH_COMPLETION,
        only_context=False,
        dataset_id=dataset_id or uuid4(),
        dataset_name=dataset_name,
    )


def test_graph_completion_items_carry_kind_and_text():
    payload = _graph_completion_payload()
    items = normalize_search_payload(payload)
    assert len(items) == 1
    assert isinstance(items[0], SearchResultItem)
    assert items[0].kind == SearchResultKind.GRAPH_COMPLETION.value
    assert items[0].text == "some grounded answer"


def test_graph_completion_extracts_structured_graph_citation():
    triplets = [
        _edge("node-a", "node-b", "edge-1"),
        _edge("node-b", "node-c", "edge-2"),
    ]
    payload = _graph_completion_payload(triplets=triplets)
    items = normalize_search_payload(payload)

    assert items
    citations = items[0].citations
    assert len(citations) == 1
    citation = citations[0]
    assert citation.kind == CitationKind.GRAPH.value
    # Every distinct source and target node from the triplets shows up
    # in the citation, deduplicated and order-preserving.
    assert citation.node_ids == ["node-a", "node-b", "node-c"]
    assert citation.edge_ids == ["edge-1", "edge-2"]


def test_graph_citation_dedupes_repeated_nodes_and_edges():
    triplets = [
        _edge("node-a", "node-b", "edge-1"),
        _edge("node-a", "node-b", "edge-1"),  # exact duplicate
        _edge("node-b", "node-c", "edge-2"),
    ]
    payload = _graph_completion_payload(triplets=triplets)
    items = normalize_search_payload(payload)

    citation = items[0].citations[0]
    assert citation.node_ids == ["node-a", "node-b", "node-c"]
    assert citation.edge_ids == ["edge-1", "edge-2"]


def test_graph_citation_handles_batched_triplet_lists():
    batched = [
        [_edge("node-a", "node-b", "edge-1")],
        [_edge("node-c", "node-d", "edge-2")],
    ]
    payload = _graph_completion_payload(triplets=batched)
    items = normalize_search_payload(payload)

    citation = items[0].citations[0]
    assert set(citation.node_ids) == {"node-a", "node-b", "node-c", "node-d"}
    assert set(citation.edge_ids) == {"edge-1", "edge-2"}


def test_graph_payload_without_triplets_emits_no_citations():
    payload = _graph_completion_payload(triplets=[])
    items = normalize_search_payload(payload)
    assert items[0].citations == []


def test_relevance_defaults_to_none_for_string_completions():
    # A pure completion string carries no score, so relevance stays
    # unset. Callers can compose confidence off ``derive_confidence``
    # which treats unset relevance as MEDIUM (recall hit, uncalibrated).
    payload = _graph_completion_payload()
    items = normalize_search_payload(payload)
    assert items[0].relevance is None


def test_relevance_normalized_from_scored_chunk_entry():
    # Simulate a chunk-shaped completion entry with an embedded score
    # (this is the shape CHUNKS-family search emits, exercised here as
    # a unit test even though full chunk citation wiring ships later).
    chunk_payload = SearchResultPayload(
        result_object=[],
        context=None,
        completion=[
            {
                "text": "a chunk of text",
                "score": 0.25,
                "document_name": "doc.txt",
                "id": "chunk-x",
            }
        ],
        search_type=SearchType.CHUNKS,
        only_context=False,
        dataset_id=uuid4(),
        dataset_name="test-dataset",
    )
    items = normalize_search_payload(chunk_payload)
    assert items[0].score == 0.25
    assert items[0].relevance is not None
    # Monotonic: score=0.25 must map above 0 and below 1.
    assert 0.0 < items[0].relevance < 1.0


def test_confidence_derivable_from_normalized_items():
    # The derive_confidence helper composes cleanly with the items the
    # normalizer produces; a graph completion with no calibrated
    # relevance yields MEDIUM (recall succeeded, ranking impossible).
    payload = _graph_completion_payload()
    items = normalize_search_payload(payload)
    assert derive_confidence(items) is Confidence.MEDIUM
