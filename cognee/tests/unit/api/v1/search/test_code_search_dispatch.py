from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.responses import dispatch_function as dispatch_module_export


@pytest.mark.asyncio
async def test_response_search_tool_forwards_code_query(monkeypatch):
    # Import the module explicitly because the package re-exports a same-named
    # callable in some installations.
    import importlib

    module = importlib.import_module("cognee.api.v1.responses.dispatch_function")
    search = AsyncMock(return_value=[{"operation": "impact_analysis"}])
    monkeypatch.setattr(module, "search", search)

    code_query = {"operation": "impact_analysis", "max_depth": 3}
    result = await module.handle_search(
        {
            "search_query": "CheckoutService",
            "search_type": "CODE",
            "datasets": ["product"],
            "code_query": code_query,
        },
        SimpleNamespace(id="user"),
    )

    assert result == [{"operation": "impact_analysis"}]
    assert search.await_args.kwargs["code_query"] == code_query


def test_response_search_tool_advertises_all_code_operations():
    # Keep this import used so tooling that evaluates package exports also
    # catches accidental removal of the response dispatcher.
    assert dispatch_module_export is not None

    from cognee.api.v1.responses.default_tools import DEFAULT_TOOLS

    search_tool = next(tool for tool in DEFAULT_TOOLS if tool["name"] == "search")
    operations = search_tool["parameters"]["properties"]["code_query"]["properties"]["operation"][
        "enum"
    ]

    assert operations == [
        "query_facts",
        "explore",
        "traverse",
        "find_path",
        "impact_analysis",
        "insights",
        "architecture",
        "delta",
        "capabilities",
        "search",
        "neighbors",
        "subgraph",
        "flows",
        "hubs",
        "ask",
    ]
    code_query_properties = search_tool["parameters"]["properties"]["code_query"]["properties"]
    assert code_query_properties["diagram"]["enum"] == ["mermaid", "dot"]
    assert code_query_properties["mode"]["enum"] == ["list", "show", "through"]
    assert {
        "text",
        "question",
        "seeds",
        "seed_ids",
        "depth",
        "max_edges",
        "mode",
        "symbol",
        "include_tests",
        "include_side",
        "max_seeds",
        "max_citations",
    } <= set(code_query_properties)


def test_both_tool_lists_advertise_the_retriever_operations():
    from cognee.api.v1.responses.default_tools import DEFAULT_TOOLS
    from cognee.api.v1.responses.routers.default_tools import DEFAULT_TOOLS as ROUTER_TOOLS
    from cognee.modules.retrieval.code_retriever import _OPERATIONS

    for tools in (DEFAULT_TOOLS, ROUTER_TOOLS):
        search_tool = next(tool for tool in tools if tool["name"] == "search")
        code_query = search_tool["parameters"]["properties"]["code_query"]["properties"]
        assert set(code_query["operation"]["enum"]) == _OPERATIONS
    assert (
        next(tool for tool in DEFAULT_TOOLS if tool["name"] == "search")["parameters"]
        == next(tool for tool in ROUTER_TOOLS if tool["name"] == "search")["parameters"]
    )
