from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.recall.types.SearchResultItem import SearchResultKind
from cognee.modules.search.models.SearchResultPayload import SearchResultPayload
from cognee.modules.search.types import SearchType


def test_hybrid_completion_normalizes_as_graph_completion():
    payload = SearchResultPayload(
        completion=["answer"],
        search_type=SearchType.HYBRID_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.GRAPH_COMPLETION


def test_chunk_result_exposes_provenance_metadata():
    """CHUNK results surface data_id/chunk_id/chunk_index so callers can trace
    a result back to the ingested document and the cited chunk."""
    chunk = {
        "id": "chunk-9",
        "document_id": "data-123",  # == ingested Data item id
        "document_name": "report.pdf",
        "chunk_index": 4,
        "text": "Revenue grew 12 percent.",
    }
    payload = SearchResultPayload(
        result_object=[chunk],
        search_type=SearchType.CHUNKS,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    metadata = items[0].metadata
    assert metadata["data_id"] == "data-123"
    assert metadata["chunk_id"] == "chunk-9"
    assert metadata["chunk_index"] == 4
    assert metadata["document_name"] == "report.pdf"
    # raw still preserves the full original payload.
    assert items[0].raw["document_id"] == "data-123"


def test_completion_result_has_no_provenance_metadata():
    """A plain completion string carries no chunk ids -> empty metadata."""
    payload = SearchResultPayload(
        completion=["Alexander is the Russian Emperor."],
        search_type=SearchType.GRAPH_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert items[0].metadata == {}
