"""FEELING_LUCKY's selector is bounded by the types that can serve the request."""

import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.search.types import SearchType

# The package re-exports the function under the module's name; patch the module.
mod = importlib.import_module("cognee.modules.search.operations.select_search_type")


def _llm_answering(answer: str):
    return patch.object(
        mod.LLMGateway, "acreate_structured_output", new=AsyncMock(return_value=answer)
    )


@pytest.mark.asyncio
async def test_choice_inside_the_allowed_set_is_kept():
    with _llm_answering("CHUNKS"):
        chosen = await mod.select_search_type(
            "q", allowed_search_types=frozenset({SearchType.CHUNKS, SearchType.RAG_COMPLETION})
        )
    assert chosen is SearchType.CHUNKS


@pytest.mark.asyncio
async def test_choice_outside_the_allowed_set_falls_to_rag_completion():
    with _llm_answering("SUMMARIES"):
        chosen = await mod.select_search_type(
            "q", allowed_search_types=frozenset({SearchType.CHUNKS, SearchType.RAG_COMPLETION})
        )
    assert chosen is SearchType.RAG_COMPLETION


@pytest.mark.asyncio
async def test_fallback_is_graph_completion_when_rag_cannot_serve_the_request():
    with _llm_answering("SUMMARIES"):
        chosen = await mod.select_search_type(
            "q", allowed_search_types=frozenset({SearchType.GRAPH_COMPLETION})
        )
    assert chosen is SearchType.GRAPH_COMPLETION


@pytest.mark.asyncio
async def test_without_a_bound_any_type_but_code_is_kept():
    with _llm_answering("SUMMARIES"):
        assert await mod.select_search_type("q") is SearchType.SUMMARIES
    with _llm_answering("CODE"):
        assert await mod.select_search_type("q") is SearchType.RAG_COMPLETION
