from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.utils.global_context import (
    format_global_context_prelude,
    search_top_global_context_summaries,
)


def test_format_global_context_prelude_returns_empty_when_both_inputs_empty():
    assert format_global_context_prelude(None, []) == ""


def test_format_global_context_prelude_omits_world_when_root_missing():
    prelude = format_global_context_prelude(None, ["area one", "area two"])
    assert prelude.startswith("Relevant areas:")
    assert "World summary" not in prelude
    assert "area one" in prelude and "area two" in prelude


def test_format_global_context_prelude_omits_areas_when_top_summaries_empty():
    prelude = format_global_context_prelude("dataset root text", [])
    assert prelude == "World summary:\ndataset root text"


def test_format_global_context_prelude_includes_both_blocks_when_both_present():
    prelude = format_global_context_prelude("dataset root text", ["area one"])
    assert prelude == "World summary:\ndataset root text\n\nRelevant areas:\narea one"


@pytest.mark.asyncio
async def test_search_top_global_context_summaries_returns_empty_when_top_k_zero():
    vector_engine = SimpleNamespace(search=AsyncMock())
    result = await search_top_global_context_summaries("query", 0, vector_engine)
    assert result == []
    vector_engine.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_top_global_context_summaries_drops_root_results():
    vector_engine = SimpleNamespace(
        search=AsyncMock(
            return_value=[
                SimpleNamespace(payload={"text": "root summary", "is_root": True}),
                SimpleNamespace(payload={"text": "area A", "is_root": False}),
                SimpleNamespace(payload={"text": "area B", "is_root": False}),
            ]
        )
    )

    result = await search_top_global_context_summaries("query", 2, vector_engine)

    assert result == ["area A", "area B"]


@pytest.mark.asyncio
async def test_search_top_global_context_summaries_skips_missing_text():
    vector_engine = SimpleNamespace(
        search=AsyncMock(
            return_value=[
                SimpleNamespace(payload={"is_root": False}),
                SimpleNamespace(payload={"text": "area A", "is_root": False}),
            ]
        )
    )

    result = await search_top_global_context_summaries("query", 5, vector_engine)

    assert result == ["area A"]


@pytest.mark.asyncio
async def test_search_top_global_context_summaries_caps_at_top_k():
    vector_engine = SimpleNamespace(
        search=AsyncMock(
            return_value=[
                SimpleNamespace(payload={"text": f"area {i}", "is_root": False}) for i in range(5)
            ]
        )
    )

    result = await search_top_global_context_summaries("query", 2, vector_engine)

    assert result == ["area 0", "area 1"]
