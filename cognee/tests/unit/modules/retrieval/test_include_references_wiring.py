"""Unit tests for the include_references wiring in the completion retrievers.

These verify the contracted append rules without any LLM or network:
- include_references=False preserves the old answer text exactly (no Evidence).
- include_references=True appends an answer-grounded Evidence block to str answers.
- Non-str response_model outputs are never corrupted.
- Evidence degrades to no-op when the backend fails or nothing overlaps the answer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


MOCK_ANSWER = "Revenue grew 12 percent."


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
        # Shares terms with MOCK_ANSWER so answer-grounded filtering keeps it.
        "text": "Revenue grew 12 percent year over year.",
    }
    return obj


@pytest.mark.asyncio
async def test_completion_references_disabled_preserves_answer():
    """include_references=False -> answer text is returned verbatim."""
    retriever = CompletionRetriever(include_references=False)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value=MOCK_ANSWER,
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert completion == [MOCK_ANSWER]


@pytest.mark.asyncio
async def test_completion_references_default_off():
    """The default is include_references=False: answers stay untouched."""
    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value=MOCK_ANSWER,
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert completion == [MOCK_ANSWER]


@pytest.mark.asyncio
async def test_completion_references_enabled_appends_evidence():
    """include_references=True -> Evidence block appended to the str answer."""
    retriever = CompletionRetriever(include_references=True)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value=MOCK_ANSWER,
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
    assert completion[0].startswith(f"{MOCK_ANSWER}\n\nEvidence:\n")
    assert "- chunk 1 of document report.pdf:" in completion[0]


@pytest.mark.asyncio
async def test_completion_references_omitted_when_answer_does_not_overlap():
    """Chunks sharing no terms with the answer are not presented as provenance."""
    retriever = CompletionRetriever(include_references=True)

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Penguins live in Antarctica.",
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context(
            "q", [_chunk_scored()], context="ctx"
        )

    assert completion == ["Penguins live in Antarctica."]


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
            return_value=MOCK_ANSWER,
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.CacheConfig",
            return_value=_cache_disabled(),
        ),
    ):
        completion = await retriever.get_completion_from_context("q", [bare], context="ctx")

    assert completion == [MOCK_ANSWER]


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
# GraphCompletionRetriever (answer-grounded chunk evidence)
# ---------------------------------------------------------------------------


def _patch_vector_engine(engine):
    return patch(
        "cognee.infrastructure.databases.vector.get_vector_engine",
        return_value=engine,
    )


@pytest.mark.asyncio
async def test_graph_references_disabled_preserves_answer():
    """include_references=False -> graph answer returned verbatim, engine never queried."""
    retriever = GraphCompletionRetriever(include_references=False)
    engine = AsyncMock()

    with _patch_vector_engine(engine):
        completion = await retriever._append_graph_evidence(["Graph answer"])

    assert completion == ["Graph answer"]
    engine.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_references_enabled_appends_answer_grounded_evidence():
    """include_references=True -> the answer is vector-queried against the chunk index."""
    retriever = GraphCompletionRetriever(include_references=True)

    engine = AsyncMock()
    engine.search.return_value = [
        MagicMock(
            id="chunk-1",
            payload={
                "document_name": "report.pdf",
                "chunk_index": 2,
                "text": "Revenue grew 12 percent year over year.",
            },
        )
    ]

    with _patch_vector_engine(engine):
        completion = await retriever._append_graph_evidence([MOCK_ANSWER])

    assert len(completion) == 1
    assert completion[0].startswith(f"{MOCK_ANSWER}\n\nEvidence:\n")
    assert "- chunk 3 of document report.pdf:" in completion[0]
    # The answer text itself is the vector query.
    assert engine.search.await_args.args[1] == MOCK_ANSWER


@pytest.mark.asyncio
async def test_graph_references_degrade_on_backend_failure():
    """A failing chunk-index search -> Evidence omitted, no raise."""
    retriever = GraphCompletionRetriever(include_references=True)

    engine = AsyncMock()
    engine.search.side_effect = RuntimeError("collection not found")

    with _patch_vector_engine(engine):
        completion = await retriever._append_graph_evidence(["Graph answer"])

    assert completion == ["Graph answer"]


@pytest.mark.asyncio
async def test_graph_references_omitted_when_nothing_overlaps_answer():
    """Vector hits unrelated to the answer are not presented as provenance."""
    retriever = GraphCompletionRetriever(include_references=True)

    engine = AsyncMock()
    engine.search.return_value = [
        MagicMock(
            id="chunk-1",
            payload={
                "document_name": "report.pdf",
                "chunk_index": 0,
                "text": "Penguins live in Antarctica.",
            },
        )
    ]

    with _patch_vector_engine(engine):
        completion = await retriever._append_graph_evidence([MOCK_ANSWER])

    assert completion == [MOCK_ANSWER]


@pytest.mark.asyncio
async def test_graph_references_skip_non_str_completions():
    """Non-str completions are never corrupted with an Evidence string."""
    retriever = GraphCompletionRetriever(include_references=True)
    structured = {"answer": "structured"}
    engine = AsyncMock()

    with _patch_vector_engine(engine):
        completion = await retriever._append_graph_evidence([structured])

    assert completion == [structured]
    engine.search.assert_not_awaited()
