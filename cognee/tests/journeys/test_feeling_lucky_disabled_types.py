"""FEELING_LUCKY answers through an enabled retriever when Cypher is disabled."""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.encoders import jsonable_encoder

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.search.types import SearchType


@pytest.mark.journey
@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["search", "recall"])
@pytest.mark.parametrize("only_context", [True, False])
@pytest.mark.parametrize("search_type", [SearchType.CYPHER, SearchType.NATURAL_LANGUAGE])
async def test_feeling_lucky_rejects_a_disabled_llm_choice(
    clean_env, default_user, tmp_path, monkeypatch, api, only_context, search_type
):
    monkeypatch.setenv("ALLOW_CYPHER_QUERY", "false")
    dataset = "journey_lucky_disabled"
    document = tmp_path / "maintenance.txt"
    document.write_text("Volvo C70 maintenance includes an annual inspection.")
    await cognee.add(str(document), dataset_name=dataset, user=default_user)
    await cognee.cognify(datasets=[dataset], user=default_user)

    original_gateway = LLMGateway.acreate_structured_output
    selector_prompt = read_query_prompt("search_type_selector_prompt.txt")
    selector = AsyncMock(return_value=search_type.value)

    async def gateway(*args, **kwargs):
        if kwargs.get("system_prompt") == selector_prompt:
            return await selector(*args, **kwargs)
        return await original_gateway(*args, **kwargs)

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", staticmethod(gateway))

    results = await getattr(cognee, api)(
        query_text="What maintenance does Volvo C70 need?",
        query_type=SearchType.FEELING_LUCKY,
        datasets=[dataset],
        only_context=only_context,
        verbose=True,
        user=default_user,
    )

    selector.assert_awaited_once()
    assert results
    assert "annual inspection" in json.dumps(jsonable_encoder(results)).lower()
    if api == "recall":
        assert all(result.search_type == SearchType.RAG_COMPLETION for result in results)
