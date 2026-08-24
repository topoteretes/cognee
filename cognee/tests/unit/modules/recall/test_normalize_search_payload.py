from pydantic import BaseModel, Field

from cognee.modules.recall.methods.normalize_search_payload import normalize_search_payload
from cognee.modules.retrieval.context_preview import CONTEXT_FORMAT_PROMPT
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


def test_code_result_preserves_structured_operation_payload():
    result = {"operation": "find_path", "found": True, "path": [{"name": "main"}]}
    payload = SearchResultPayload(completion=result, search_type=SearchType.CODE)

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.CODE
    assert items[0].raw == result


def test_agentic_completion_normalizes_as_graph_completion():
    payload = SearchResultPayload(
        completion=["Agentic answer"],
        search_type=SearchType.AGENTIC_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.GRAPH_COMPLETION
    assert items[0].search_type == SearchType.AGENTIC_COMPLETION


def test_feeling_lucky_resolved_type_normalizes_with_matching_kind():
    """Payload should carry the resolved retriever type (not FEELING_LUCKY)."""
    payload = SearchResultPayload(
        result_object=[{"id": "chunk-1", "text": "snippet", "document_id": "doc-1"}],
        search_type=SearchType.CHUNKS,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.CHUNK
    assert items[0].search_type == SearchType.CHUNKS


class _AnalysisResult(BaseModel):
    answer: str = Field(..., description="Short answer")
    confidence: float = Field(..., description="0-1 confidence")


def test_structured_response_model_populates_structured_field():
    model = _AnalysisResult(answer="Revenue grew 12%", confidence=0.91)
    payload = SearchResultPayload(
        completion=[model],
        search_type=SearchType.GRAPH_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    assert items[0].kind == SearchResultKind.STRUCTURED
    assert items[0].structured == {"answer": "Revenue grew 12%", "confidence": 0.91}
    assert items[0].raw == items[0].structured
    assert "Revenue grew 12%" in items[0].text


def test_only_context_default_format_yields_one_item_per_context_entry():
    payload = SearchResultPayload(
        context=["triplet-a", "triplet-b"],
        only_context=True,
        search_type=SearchType.GRAPH_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert [item.text for item in items] == ["triplet-a", "triplet-b"]


def test_only_context_prompt_format_yields_one_item_carrying_the_parts():
    """The prompt is a single artifact, so it must not be split per context entry."""
    payload = SearchResultPayload(
        context=["triplet-a", "triplet-b"],
        only_context=True,
        context_format=CONTEXT_FORMAT_PROMPT,
        question="why?",
        session_context="## Active session guidance\n- be terse",
        user_prompt="The question is: `why?` ... triplet-a\n---\ntriplet-b",
        system_prompt="history\nTASK:answer",
        search_type=SearchType.GRAPH_COMPLETION,
    )

    items = normalize_search_payload(payload)

    assert len(items) == 1
    item = items[0]
    assert item.text == "The question is: `why?` ... triplet-a\n---\ntriplet-b"
    # The parts stay addressable so a caller can still take just one layer.
    assert item.raw["question"] == "why?"
    assert item.raw["context"] == ["triplet-a", "triplet-b"]
    assert item.raw["session_context"] == "## Active session guidance\n- be terse"
    assert item.raw["system_prompt"] == "history\nTASK:answer"
