"""Unit tests for structured citation extraction.

Covers ``build_chunk_citations`` (structured parallel to
``format_chunk_references``): shape of emitted :class:`Citation`
objects, dedup, answer-overlap filtering, and the 3-5 cap.
"""

from __future__ import annotations

from cognee.modules.retrieval.utils.citation_models import Citation, CitationKind
from cognee.modules.retrieval.utils.references import build_chunk_citations


def _chunk(
    document_name: str = "doc.txt",
    chunk_number: int = 1,
    text: str = "quick brown fox jumps over the lazy dog",
    chunk_id: str | None = "chunk-1",
    document_id: str | None = "doc-1",
) -> dict:
    return {
        "payload": {
            "document_name": document_name,
            "chunk_number": chunk_number,
            "text": text,
            "id": chunk_id,
            "document_id": document_id,
        },
        "id": chunk_id,
    }


def test_empty_input_returns_empty_list():
    assert build_chunk_citations([]) == []
    assert build_chunk_citations(None) == []


def test_chunk_without_document_name_is_skipped():
    citations = build_chunk_citations([_chunk(document_name="")])
    assert citations == []


def test_chunk_missing_number_is_skipped():
    payload = _chunk()
    payload["payload"]["chunk_number"] = None
    payload["payload"]["chunk_index"] = None
    citations = build_chunk_citations([payload])
    assert citations == []


def test_emitted_citation_carries_all_expected_fields():
    citation_list = build_chunk_citations([_chunk()])
    assert len(citation_list) == 1
    citation = citation_list[0]
    assert isinstance(citation, Citation)
    assert citation.kind == CitationKind.CHUNK.value
    assert citation.document_name == "doc.txt"
    assert citation.chunk_number == 1
    assert citation.chunk_id == "chunk-1"
    assert citation.data_id == "doc-1"
    assert citation.snippet is not None
    assert "quick brown fox" in citation.snippet


def test_duplicate_chunks_are_deduplicated():
    same_chunk = [_chunk(chunk_id="chunk-1"), _chunk(chunk_id="chunk-1")]
    citations = build_chunk_citations(same_chunk)
    assert len(citations) == 1


def test_answer_overlap_filters_chunks_without_shared_terms():
    payloads = [
        _chunk(text="the quick brown fox"),
        _chunk(chunk_number=2, chunk_id="chunk-2", text="astronomy planets galaxy"),
    ]
    citations = build_chunk_citations(payloads, answer="A quick fox appeared")
    # Only the chunk sharing 'quick' / 'fox' with the answer survives.
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-1"


def test_answer_overlap_orders_by_shared_term_count():
    payloads = [
        _chunk(chunk_id="low", text="fox"),
        _chunk(chunk_number=2, chunk_id="high", text="quick brown fox"),
    ]
    answer = "the quick brown fox"
    citations = build_chunk_citations(payloads, answer=answer)
    assert [c.chunk_id for c in citations] == ["high", "low"]


def test_answer_with_no_significant_terms_returns_empty():
    citations = build_chunk_citations([_chunk()], answer="yes.")
    assert citations == []


def test_limit_is_clamped_into_three_to_five_range():
    many = [_chunk(chunk_number=index, chunk_id=f"chunk-{index}") for index in range(1, 11)]
    # Requesting fewer than 3 clamps up to 3; more than 5 clamps to 5.
    assert len(build_chunk_citations(many, limit=1)) == 3
    assert len(build_chunk_citations(many, limit=100)) == 5


def test_missing_data_id_is_none_not_empty_string():
    payload = _chunk()
    payload["payload"]["document_id"] = None
    citations = build_chunk_citations([payload])
    assert citations[0].data_id is None


def test_snippet_is_truncated_when_source_is_long():
    long_text = "sentence " * 100
    citation = build_chunk_citations([_chunk(text=long_text)])[0]
    assert citation.snippet is not None
    # Truncation marker plus bounded length keep the citation usable
    # for a UI even when the underlying chunk is huge.
    assert citation.snippet.endswith("…") or len(citation.snippet) < len(long_text)
