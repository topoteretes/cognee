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


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE])
@pytest.mark.parametrize("flag", [None, "false"])
@pytest.mark.parametrize("graph_only", [False, True])
async def test_disabled_cypher_choice_uses_an_allowed_default(
    monkeypatch, search_type, flag, graph_only
):
    if flag is None:
        monkeypatch.delenv("ALLOW_CYPHER_QUERY", raising=False)
    else:
        monkeypatch.setenv("ALLOW_CYPHER_QUERY", flag)
    # Even an explicit request allowlist cannot enable a disabled type.
    allowed = frozenset({SearchType.GRAPH_COMPLETION, search_type}) if graph_only else None
    with _llm_answering(search_type.value):
        chosen = await mod.select_search_type("q", allowed_search_types=allowed)
    expected = SearchType.GRAPH_COMPLETION if graph_only else SearchType.RAG_COMPLETION
    assert chosen is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE])
@pytest.mark.parametrize("flag", ["true", "1", "yes", " TRUE "])
async def test_enabled_cypher_choice_is_kept(monkeypatch, search_type, flag):
    monkeypatch.setenv("ALLOW_CYPHER_QUERY", flag)
    with _llm_answering(search_type.value):
        assert await mod.select_search_type("q") is search_type


@pytest.mark.asyncio
@pytest.mark.parametrize("search_type", [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE])
async def test_enabled_cypher_choice_still_respects_request_scope(monkeypatch, search_type):
    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "true")
    with _llm_answering(search_type.value):
        chosen = await mod.select_search_type(
            "q", allowed_search_types=frozenset({SearchType.GRAPH_COMPLETION})
        )
    assert chosen is SearchType.GRAPH_COMPLETION
