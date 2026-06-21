"""Unit tests for the LLM-free reference (Evidence) helpers.

Covers ``format_chunk_references`` (sync, payload-driven, answer-grounded) and
``build_answer_grounded_chunk_references`` (async, vector-engine-driven),
including the old-data graceful-degradation cases and backend-failure cases.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.utils.references import (
    EVIDENCE_HEADER,
    build_answer_grounded_chunk_references,
    format_chunk_references,
)


# ---------------------------------------------------------------------------
# format_chunk_references (no answer: legacy retrieval-order behavior)
# ---------------------------------------------------------------------------


def _payload(**overrides):
    base = {
        "document_name": "annual_report.pdf",
        "chunk_index": 4,  # 1-based display number -> 5
        "text": "Revenue grew 12 percent year over year.",
    }
    base.update(overrides)
    return base


def test_format_chunk_references_renders_one_based_number_from_chunk_index():
    """chunk_index + 1 is rendered as the display number."""
    result = format_chunk_references([_payload(chunk_index=4)])

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- chunk 5 of document annual_report.pdf:" in result
    assert "Revenue grew 12 percent" in result


def test_format_chunk_references_prefers_explicit_chunk_number():
    """An explicit chunk_number wins over chunk_index when both present."""
    result = format_chunk_references([_payload(chunk_number=9, chunk_index=4)])

    assert "- chunk 9 of document annual_report.pdf:" in result
    assert "chunk 5" not in result


def test_format_chunk_references_reads_scored_result_like_objects():
    """ScoredResult-like objects expose .payload and .id; they are read correctly."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [FakeScored(_payload(), "id-1")]
    result = format_chunk_references(objs)

    assert "- chunk 5 of document annual_report.pdf:" in result


def test_format_chunk_references_empty_when_document_name_missing():
    """Old-data case: missing document_name -> entry skipped -> empty string."""
    payload = _payload()
    del payload["document_name"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_document_name_null():
    """Null document_name is unusable -> empty string."""
    assert format_chunk_references([_payload(document_name=None)]) == ""


def test_format_chunk_references_empty_when_no_chunk_number():
    """Missing both chunk_number and chunk_index -> empty string."""
    payload = _payload()
    del payload["chunk_index"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_when_text_missing():
    """Missing text -> no usable snippet -> empty string."""
    payload = _payload()
    del payload["text"]

    assert format_chunk_references([payload]) == ""


def test_format_chunk_references_empty_for_empty_input():
    assert format_chunk_references([]) == ""
    assert format_chunk_references(None) == ""


def test_format_chunk_references_dedups_by_id():
    """Two payloads with the same object id collapse into one bullet."""

    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    objs = [
        FakeScored(_payload(text="first"), "same-id"),
        FakeScored(_payload(text="second"), "same-id"),
    ]
    result = format_chunk_references(objs)

    assert result.count("- chunk 5 of document annual_report.pdf:") == 1


def test_format_chunk_references_caps_and_clamps_limit():
    """Limit is clamped into the 3-5 range."""
    payloads = [
        _payload(document_name=f"doc_{i}.pdf", chunk_index=i, text=f"text {i}", id=str(i))
        for i in range(10)
    ]
    # limit below the floor is clamped up to 3
    low = format_chunk_references(payloads, limit=1)
    assert low.count("- chunk ") == 3
    # limit above the ceiling is clamped down to 5
    high = format_chunk_references(payloads, limit=99)
    assert high.count("- chunk ") == 5


def test_format_chunk_references_snippet_truncated():
    """Long text is truncated with an ellipsis."""
    long_text = "word " * 100
    result = format_chunk_references([_payload(text=long_text)])
    # The bullet line contains a truncation ellipsis.
    assert "…" in result


# ---------------------------------------------------------------------------
# format_chunk_references (answer-grounded filtering and ranking)
# ---------------------------------------------------------------------------


def test_answer_filtering_drops_chunks_without_overlap():
    """Chunks sharing no significant terms with the answer are not cited."""
    matching = _payload(document_name="report.pdf", chunk_index=0, text="Revenue grew 12 percent.")
    unrelated = _payload(
        document_name="other.pdf", chunk_index=1, text="Penguins live in Antarctica."
    )

    result = format_chunk_references([matching, unrelated], answer="Revenue grew 12 percent.")

    assert "report.pdf" in result
    assert "other.pdf" not in result


def test_answer_filtering_empty_when_nothing_overlaps():
    """No candidate overlaps the answer -> Evidence omitted entirely."""
    unrelated = _payload(text="Penguins live in Antarctica.")

    assert format_chunk_references([unrelated], answer="Quarterly revenue increased.") == ""


def test_answer_filtering_ranks_by_overlap():
    """Higher answer-term overlap is cited before lower overlap."""
    weak = _payload(document_name="weak.pdf", chunk_index=0, text="Revenue is mentioned once.")
    strong = _payload(
        document_name="strong.pdf",
        chunk_index=1,
        text="Revenue grew twelve percent in the fourth quarter.",
    )

    result = format_chunk_references(
        [weak, strong], answer="Revenue grew twelve percent in the fourth quarter."
    )

    assert result.index("strong.pdf") < result.index("weak.pdf")


def test_answer_with_no_significant_terms_yields_no_evidence():
    """An answer made of stopwords/stubs cannot be grounded -> empty string."""
    assert format_chunk_references([_payload()], answer="It is.") == ""


def test_answer_none_keeps_all_usable_candidates():
    """answer=None preserves the unfiltered retrieval-order behavior."""
    unrelated = _payload(text="Penguins live in Antarctica.")

    assert "Penguins" in format_chunk_references([unrelated], answer=None)


# ---------------------------------------------------------------------------
# build_answer_grounded_chunk_references
# ---------------------------------------------------------------------------


def _scored(payload, id_):
    class FakeScored:
        def __init__(self, payload, id_):
            self.payload = payload
            self.id = id_

    return FakeScored(payload, id_)


@pytest.mark.asyncio
async def test_answer_grounded_references_query_chunk_index_with_answer():
    """The answer text is run as the vector query and grounds the bullets."""
    engine = AsyncMock()
    engine.search.return_value = [
        _scored(
            _payload(document_name="report.pdf", chunk_index=2, text="Revenue grew 12 percent."),
            "chunk-1",
        )
    ]

    result = await build_answer_grounded_chunk_references("Revenue grew 12 percent.", engine)

    assert result.startswith(EVIDENCE_HEADER + "\n")
    assert "- chunk 3 of document report.pdf:" in result
    engine.search.assert_awaited_once()
    assert engine.search.await_args.args[0] == "DocumentChunk_text"
    assert engine.search.await_args.args[1] == "Revenue grew 12 percent."


@pytest.mark.asyncio
async def test_answer_grounded_references_drop_unrelated_results():
    """Vector hits that share no terms with the answer are filtered out."""
    engine = AsyncMock()
    engine.search.return_value = [
        _scored(_payload(text="Penguins live in Antarctica."), "chunk-1"),
    ]

    result = await build_answer_grounded_chunk_references("Quarterly revenue increased.", engine)

    assert result == ""


@pytest.mark.asyncio
async def test_answer_grounded_references_empty_on_search_failure():
    """A missing collection or backend failure degrades to no Evidence, no raise."""
    engine = AsyncMock()
    engine.search.side_effect = RuntimeError("collection not found")

    result = await build_answer_grounded_chunk_references("Revenue grew.", engine)

    assert result == ""


@pytest.mark.asyncio
async def test_answer_grounded_references_empty_for_blank_answer_or_engine():
    assert await build_answer_grounded_chunk_references("", AsyncMock()) == ""
    assert await build_answer_grounded_chunk_references("   ", AsyncMock()) == ""
    assert await build_answer_grounded_chunk_references("Revenue grew.", None) == ""
