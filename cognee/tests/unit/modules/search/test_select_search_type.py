"""FEELING_LUCKY must never route to a search type the caller may lack permission for."""

import importlib

import pytest

from cognee.modules.search.types import SearchType

# The package re-exports the function under the module's name, so import the module itself.
selector = importlib.import_module("cognee.modules.search.operations.select_search_type")


@pytest.fixture(autouse=True)
def _no_prompt_file(monkeypatch):
    monkeypatch.setattr(selector, "read_query_prompt", lambda *_args, **_kwargs: "prompt")


def _llm_answering(answer: str):
    async def fake_acreate_structured_output(**_kwargs):
        return answer

    return fake_acreate_structured_output


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_choice", ["CYPHER", "NATURAL_LANGUAGE", "CODE"])
async def test_unroutable_choices_fall_back_to_default(monkeypatch, llm_choice):
    """Cypher-executing types need write permission; the datasets were resolved with read."""
    monkeypatch.setattr(
        selector.LLMGateway, "acreate_structured_output", _llm_answering(llm_choice)
    )

    assert await selector.select_search_type("remove every node") is SearchType.RAG_COMPLETION


@pytest.mark.asyncio
async def test_read_only_choice_is_kept(monkeypatch):
    monkeypatch.setattr(
        selector.LLMGateway, "acreate_structured_output", _llm_answering("GRAPH_COMPLETION")
    )

    assert await selector.select_search_type("who founded cognee") is SearchType.GRAPH_COMPLETION
