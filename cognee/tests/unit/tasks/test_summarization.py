"""
Unit tests for summarization tasks with mocked LLM.

Validates summarize_text and related logic without making real LLM API calls.
"""

from uuid import uuid4

import pytest
from unittest.mock import AsyncMock, patch

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.shared.data_models import SummarizedContent
from cognee.tasks.summarization import summarize_text
from cognee.tasks.summarization.exceptions import InvalidSummaryInputsError
from cognee.tasks.summarization.models import TextSummary


def _make_document()-> TextDocument:
    """Minimal TextDocument for chunk ownership."""
    return TextDocument(
        id=uuid4(),
        name="test_doc",
        raw_data_location="/tmp/test.txt",
        external_metadata=None,
        mime_type="text/plain",
    )


def _make_chunk(text: str, chunk_index: int = 0) -> DocumentChunk:
    """Create a DocumentChunk with required fields."""
    doc = _make_document()
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_index=chunk_index,
        chunk_size=len(text),
        cut_type="sentence_end",
        is_part_of=doc,
    )


@pytest.mark.asyncio
async def test_summarize_text_basic_returns_text_summaries()-> None:
    """Basic summarization: single chunk, mocked LLM returns summary."""
    chunk = _make_chunk("Cognee turns documents into AI memory for agents.")
    mock_summary = SummarizedContent(summary="Summary here", description="Short desc")

    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new_callable=AsyncMock,
        return_value=mock_summary,
    ):
        result = await summarize_text([chunk], summarization_model=SummarizedContent)

    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], TextSummary)
    assert result[0].text == "Summary here"
    assert result[0].made_from is chunk


@pytest.mark.asyncio
async def test_summarize_text_empty_chunks_returns_unchanged() -> None:
    """Empty list of chunks returns immediately without calling LLM."""
    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new_callable=AsyncMock,
    ) as mock_extract:
        result = await summarize_text([], summarization_model=SummarizedContent)

    assert result == []
    mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_text_short_text_still_summarized()-> None:
    """Very short or empty chunk text still goes through summarization (mocked)."""
    chunk = _make_chunk("")
    mock_summary = SummarizedContent(summary="Empty summary", description="N/A")

    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new_callable=AsyncMock,
        return_value=mock_summary,
    ):
        result = await summarize_text([chunk], summarization_model=SummarizedContent)

    assert len(result) == 1
    assert result[0].text == "Empty summary"
    assert result[0].made_from is chunk


@pytest.mark.asyncio
async def test_summarize_text_multiple_chunks()-> None:
    """Multiple chunks get summarized and mapped correctly."""
    chunk1 = _make_chunk("First paragraph about AI.", 0)
    chunk2 = _make_chunk("Second paragraph about memory.", 1)
    chunk3 = _make_chunk("Third paragraph about graphs.", 2)

    summaries = [
        SummarizedContent(summary="First summary", description="d1"),
        SummarizedContent(summary="Second summary", description="d2"),
        SummarizedContent(summary="Third summary", description="d3"),
    ]

    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new_callable=AsyncMock,
        side_effect=summaries,
    ):
        result = await summarize_text(
            [chunk1, chunk2, chunk3],
            summarization_model=SummarizedContent,
        )

    assert len(result) == 3
    assert result[0].text == "First summary"
    assert result[0].made_from is chunk1
    assert result[1].text == "Second summary"
    assert result[1].made_from is chunk2
    assert result[2].text == "Third summary"
    assert result[2].made_from is chunk3


@pytest.mark.asyncio
async def test_summarize_text_llm_failure_propagates()-> None:
    """When extract_summary raises, the exception propagates."""
    chunk = _make_chunk("Some text.")
    with patch(
        "cognee.tasks.summarization.summarize_text.extract_summary",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM call failed"),
    ):
        with pytest.raises(RuntimeError, match="LLM call failed"):
            await summarize_text([chunk], summarization_model=SummarizedContent)


@pytest.mark.asyncio
async def test_summarize_text_validates_list_input()-> None:
    """Non-list data_chunks raises InvalidSummaryInputsError."""
    with pytest.raises(InvalidSummaryInputsError, match="data_chunks must be a list"):
        await summarize_text("not a list", summarization_model=SummarizedContent)


@pytest.mark.asyncio
async def test_summarize_text_validates_chunk_has_text()-> None:
    """Chunks without 'text' attribute raise InvalidSummaryInputsError."""
    bad_chunk = type("BadChunk", (), {"id": uuid4()})()  # no .text
    with pytest.raises(
        InvalidSummaryInputsError,
        match="each DocumentChunk must have a 'text' attribute",
    ):
        await summarize_text([bad_chunk], summarization_model=SummarizedContent)


@pytest.mark.asyncio
async def test_summarize_text_uses_default_model_when_none()-> None:
    """When summarization_model is None, get_cognify_config is used."""
    chunk = _make_chunk("Config-driven model test.")
    mock_summary = SummarizedContent(summary="From default model", description="desc")

    with (
        patch(
            "cognee.tasks.summarization.summarize_text.extract_summary",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ),
        patch(
            "cognee.tasks.summarization.summarize_text.get_cognify_config",
        ) as mock_config,
    ):
        mock_config.return_value.summarization_model = SummarizedContent
        result = await summarize_text([chunk], summarization_model=None)

    assert len(result) == 1
    assert result[0].text == "From default model"
    mock_config.assert_called_once()
