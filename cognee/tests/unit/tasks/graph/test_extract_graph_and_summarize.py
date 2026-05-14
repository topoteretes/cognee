import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

egs_module = importlib.import_module("cognee.tasks.graph.extract_graph_and_summarize")


def _make_chunk(text: str, index: int = 0) -> DocumentChunk:
    document = TextDocument(
        name="document.txt",
        raw_data_location="document.txt",
        external_metadata=None,
    )
    return DocumentChunk(
        text=text,
        chunk_size=len(text),
        chunk_index=index,
        cut_type="test",
        is_part_of=document,
    )


def _make_graph(node_id: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                id=node_id,
                name=node_id,
                type="Entity",
                description=f"{node_id} description",
            )
        ],
        edges=[],
    )


@pytest.mark.asyncio
@patch.object(egs_module, "summarize_text", new_callable=AsyncMock)
@patch.object(egs_module, "extract_graph_from_data", new_callable=AsyncMock)
@patch.object(egs_module, "extract_content_graph_and_summary", new_callable=AsyncMock)
async def test_extract_graph_and_summarize_uses_combined_llm_extraction(
    mock_extract_content_graph_and_summary,
    mock_extract_graph_from_data,
    mock_summarize_text,
):
    chunks = [_make_chunk("Alice knows Bob.", 0), _make_chunk("Bob works at Cognee.", 1)]
    graphs = [_make_graph("Alice"), _make_graph("Bob")]

    mock_extract_content_graph_and_summary.side_effect = [
        SimpleNamespace(
            graph=graphs[0],
            summary=SummarizedContent(summary="Alice knows Bob.", description="Relationship"),
        ),
        SimpleNamespace(
            graph=graphs[1],
            summary=SummarizedContent(summary="Bob works at Cognee.", description="Work"),
        ),
    ]

    async def integrate_with_precomputed_graphs(*, data_chunks, calculate_chunk_graphs, **kwargs):
        assert await calculate_chunk_graphs(data_chunks) == graphs
        return data_chunks

    mock_extract_graph_from_data.side_effect = integrate_with_precomputed_graphs

    result = await egs_module.extract_graph_and_summarize(
        data_chunks=chunks,
        graph_model=KnowledgeGraph,
    )

    assert mock_extract_content_graph_and_summary.await_count == 2
    assert [call.args[0] for call in mock_extract_content_graph_and_summary.await_args_list] == [
        "Alice knows Bob.",
        "Bob works at Cognee.",
    ]
    mock_extract_graph_from_data.assert_awaited_once()
    mock_summarize_text.assert_not_awaited()
    assert [summary.text for summary in result] == [
        "Alice knows Bob.",
        "Bob works at Cognee.",
    ]
    assert [summary.made_from for summary in result] == chunks


@pytest.mark.asyncio
@patch.object(egs_module, "extract_content_graph_and_summary", new_callable=AsyncMock)
@patch.object(egs_module, "summarize_text", new_callable=AsyncMock)
@patch.object(egs_module, "extract_graph_from_data", new_callable=AsyncMock)
async def test_extract_graph_and_summarize_preserves_custom_graph_hook_path(
    mock_extract_graph_from_data,
    mock_summarize_text,
    mock_extract_content_graph_and_summary,
):
    chunks = [_make_chunk("Alice knows Bob.")]
    summaries = [SimpleNamespace(text="legacy summary")]

    mock_extract_graph_from_data.return_value = chunks
    mock_summarize_text.return_value = summaries

    result = await egs_module.extract_graph_and_summarize(
        data_chunks=chunks,
        graph_model=KnowledgeGraph,
        calculate_chunk_graphs=lambda *_args, **_kwargs: [],
    )

    assert result == summaries
    mock_extract_graph_from_data.assert_awaited_once()
    mock_summarize_text.assert_awaited_once()
    mock_extract_content_graph_and_summary.assert_not_awaited()
