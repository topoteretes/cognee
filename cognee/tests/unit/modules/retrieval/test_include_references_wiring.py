"""Unit tests for the include_references wiring in the completion retrievers.

These verify the contracted append rules without any LLM or network:
- include_references=False preserves the old answer text exactly (no Evidence).
- include_references=True appends a non-empty Evidence block to str answers.
- Non-str response_model outputs are never corrupted.
- Graph evidence degrades to no-op when the backend cannot traverse.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


def _cache_disabled():
    """Patch CacheConfig so the non-session branch is exercised."""
    mock_config = MagicMock()
    mock_config.caching = False
    return mock_config


# ---------------------------------------------------------------------------
# CompletionRetriever (RAG / chunk evidence)
# ---------------------------------------------------------------------------


def _chunk_scored():
    obj = MagicMock()
    obj.id = "chunk-1"
    obj.payload = {
        "document_name": "report.pdf",
        "chunk_index": 0,
        "text": "Relevant supporting text.",
    }
    return obj


@pytest.mark.asyncio
async def test_completion_references_disabled_preserves_answer():
    """include_references=False -> answer text is returned verbatim."""
    retriever = CompletionRetriever(include_references=False)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert completion == ["Generated answer"]


@pytest.mark.asyncio
async def test_completion_references_enabled_appends_evidence():
    """include_references=True -> Evidence block appended to the str answer."""
    retriever = CompletionRetriever(include_references=True)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert len(completion) == 1
    assert completion[0].startswith("Generated answer\n\nEvidence:\n")
    assert "- chunk 1 of document report.pdf:" in completion[0]


@pytest.mark.asyncio
async def test_completion_references_enabled_but_no_usable_payload_omits_evidence():
    """Old-data payload (no reference fields) -> Evidence omitted, answer unchanged."""
    retriever = CompletionRetriever(include_references=True)
    bare = MagicMock()
    bare.id = "x"
    bare.payload = {"text": "only text, no name/index"}

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context("q", [bare], context="ctx")

    assert completion == ["Generated answer"]


@pytest.mark.asyncio
async def test_completion_references_skipped_for_non_str_response_model():
    """Non-str response_model is never corrupted with an Evidence string."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    retriever = CompletionRetriever(include_references=True, response_model=Answer)
    model_obj = Answer(answer="structured")

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value=model_obj,
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert completion == [model_obj]


# ---------------------------------------------------------------------------
# GraphCompletionRetriever (graph entity-fallback evidence)
# ---------------------------------------------------------------------------


def _patch_graph_engine(engine):
    return patch(
        "cognee.infrastructure.databases.graph.get_graph_engine",
        new=AsyncMock(return_value=engine),
    )


@pytest.mark.asyncio
async def test_graph_references_disabled_preserves_answer():
    """include_references=False -> graph answer returned verbatim, engine never queried."""
    retriever = GraphCompletionRetriever(include_references=False)

    # node_ids resolution should not even matter, but make it non-trivial.
    with patch.object(retriever, "_node_ids_from_retrieved", return_value=["entity-1"]):
        completion = await retriever._append_graph_evidence(["Graph answer"], object())

    assert completion == ["Graph answer"]


@pytest.mark.asyncio
async def test_graph_references_enabled_appends_evidence():
    """include_references=True -> entity-fallback Evidence appended to str answer."""
    retriever = GraphCompletionRetriever(include_references=True)

    engine = AsyncMock()
    engine.get_connections.return_value = [
        (
            {"id": "entity-1", "name": "Acme Corp", "type": "Entity"},
            {"relationship_name": "contains"},
            {
                "id": "chunk-1",
                "name": "chunk-1",
                "type": "DocumentChunk",
                "chunk_index": 2,
                "document_name": "report.pdf",
            },
        )
    ]

    with (
        patch.object(retriever, "_node_ids_from_retrieved", return_value=["entity-1"]),
        _patch_graph_engine(engine),
    ):
        completion = await retriever._append_graph_evidence(["Graph answer"], object())

    assert len(completion) == 1
    assert completion[0].startswith("Graph answer\n\nEvidence:\n")
    assert "- Entity Acme Corp appears in chunk 3 of document report.pdf" in completion[0]


@pytest.mark.asyncio
async def test_graph_references_degrade_on_non_traversable_backend():
    """Postgres-graph (NotImplementedError) -> Evidence omitted, no raise."""
    retriever = GraphCompletionRetriever(include_references=True)

    engine = AsyncMock()
    engine.get_connections.side_effect = NotImplementedError("not supported")

    with (
        patch.object(retriever, "_node_ids_from_retrieved", return_value=["entity-1"]),
        _patch_graph_engine(engine),
    ):
        completion = await retriever._append_graph_evidence(["Graph answer"], object())

    assert completion == ["Graph answer"]


@pytest.mark.asyncio
async def test_graph_references_no_node_ids_omits_evidence():
    """No entity node ids resolved -> Evidence omitted, engine not consulted."""
    retriever = GraphCompletionRetriever(include_references=True)

    with patch.object(retriever, "_node_ids_from_retrieved", return_value=[]):
        completion = await retriever._append_graph_evidence(["Graph answer"], object())

    assert completion == ["Graph answer"]
