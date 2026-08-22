"""Behavior of the tool-search gating added in CLO-352.

The guarantee these tests protect: shrinking what ``tools/list`` advertises must
not shrink what is *callable*. Every unadvertised tool has to stay reachable
both directly by name and through the ``call_tool`` proxy (how an agent calls
something it just found).
"""

import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

import src.server as server  # noqa: E402
from src.tool_registry import DEFAULT_TAG, MEMORY_TAG  # noqa: E402

SYNTHETIC_TOOLS = {"search_tools", "call_tool"}
MEMORY_TOOLS = {"remember", "recall", "forget"}

# Deliberately not enumerated: the unpinned tools are derived from the registry so
# this file keeps working as the catalog changes. Only the pinned sets are spelled
# out, because those are the contract worth reviewing by eye.


def hidden_tools() -> set[str]:
    """Registered but not advertised in default mode."""
    return set(server.registry.tags) - set(server.registry.names_with_tag(DEFAULT_TAG))


class FakeStatusClient:
    """Drives cognify_status down its API-mode path with no real backend."""

    use_api = True

    async def list_datasets(self):
        return [{"id": "id-1", "name": "main_dataset"}]

    async def get_pipeline_status(self, dataset_ids, pipeline_name):
        return {"id-1": "DATASET_PROCESSING_COMPLETED"}


@pytest.fixture(autouse=True)
def restore_tool_mode():
    """Tests mutate the module-global server's transforms; put them back."""
    saved = list(server.mcp._transforms)
    yield
    server.mcp._transforms = saved


async def advertised(mode: str) -> set[str]:
    server.apply_tool_mode(mode)
    return {tool.name for tool in await server.mcp.list_tools()}


async def search(client, query: str) -> list[str]:
    result = await client.call_tool("search_tools", {"query": query})
    if not result.content:  # no match -> empty content rather than an empty array
        return []
    return [tool["name"] for tool in json.loads(result.content[0].text)]


# --- tier declarations ---------------------------------------------------------


async def test_every_tool_declares_a_tier():
    """A tool registered with a bare @mcp.tool would be invisible to
    names_with_tag(), so it would never be pinned and never be counted as hidden.

    Compared against the live catalog rather than a hardcoded list, so adding or
    removing a tool needs no edit here.
    """
    server.apply_tool_mode("all")
    advertised_names = {tool.name for tool in await server.mcp.list_tools()}

    assert set(server.registry.tags) == advertised_names
    assert all(tags for tags in server.registry.tags.values())


def test_pinned_sets_are_derived_from_tags():
    assert set(server.registry.names_with_tag(MEMORY_TAG)) == MEMORY_TOOLS
    assert set(server.registry.names_with_tag(DEFAULT_TAG)) == MEMORY_TOOLS


# --- what each mode advertises ------------------------------------------------


async def test_default_mode_advertises_pinned_plus_synthetic():
    assert await advertised("default") == MEMORY_TOOLS | SYNTHETIC_TOOLS


async def test_minimal_mode_advertises_only_the_memory_api():
    assert await advertised("minimal") == MEMORY_TOOLS | SYNTHETIC_TOOLS


async def test_all_mode_restores_the_flat_surface():
    names = await advertised("all")
    assert names == set(server.registry.tags)
    assert not names & SYNTHETIC_TOOLS


async def test_unknown_mode_falls_back_to_default():
    assert server.apply_tool_mode("banana") == "default"
    assert {tool.name for tool in await server.mcp.list_tools()} == (MEMORY_TOOLS | SYNTHETIC_TOOLS)


async def test_apply_tool_mode_is_idempotent():
    """add_transform() appends, so a second call must not stack a second search
    transform (which would hide the first one's pinned tools)."""
    await advertised("default")
    first = {tool.name for tool in await server.mcp.list_tools()}

    server.apply_tool_mode("default")
    assert {tool.name for tool in await server.mcp.list_tools()} == first

    # ...and switching modes replaces rather than layers.
    server.apply_tool_mode("minimal")
    assert {tool.name for tool in await server.mcp.list_tools()} == MEMORY_TOOLS | SYNTHETIC_TOOLS


# --- the lazy-loading guarantee ------------------------------------------------


async def test_result_window_is_not_the_binding_constraint():
    """TOOL_SEARCH_MAX_RESULTS is sized for a catalog we expect to grow, so today
    it exceeds the number of hidden tools: no hidden tool can be pushed out of the
    window by the limit alone."""
    hidden = hidden_tools()
    assert len(hidden) <= server.TOOL_SEARCH_MAX_RESULTS


@pytest.mark.parametrize(
    "query, expected",
    [
        ("is my background ingestion finished?", "cognify_status"),
        ("check the progress of a pipeline job", "cognify_status"),
        ("did remember fail in the background", "cognify_status"),
    ],
)
async def test_natural_language_queries_rank_their_tool_first(query, expected):
    """The phrasings an agent actually produces are multi-word and land at rank 1.
    Pinned here so a description edit that breaks discoverability fails loudly."""
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        results = await search(client, query)

    assert results and results[0] == expected, f"{query!r} -> {results}"


async def test_search_matches_on_vocabulary_not_everything():
    """BM25 drops zero-scoring tools: a query sharing no vocabulary with a tool's
    description returns nothing, which makes tool *descriptions* the lever for
    recall. A query covering the description's vocabulary reaches every hidden
    tool, confirming the result limit is not the cap."""
    server.apply_tool_mode("default")
    hidden = hidden_tools()

    async with Client(server.mcp) as client:
        unrelated = set(await search(client, "banana smoothie recipe"))
        combined = set(await search(client, "background ingestion pipeline status progress"))

    assert not unrelated
    assert combined == hidden


async def test_hidden_tools_are_discoverable_by_search():
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        assert "cognify_status" in await search(client, "check ingestion status")


async def test_search_never_returns_pinned_tools():
    """Pinned tools are already in tools/list; echoing them back would waste the
    result budget the search transform exists to protect."""
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        for query in ("remember this for later", "search my memory", "delete a dataset"):
            assert not MEMORY_TOOLS & set(await search(client, query))


async def test_hidden_tool_is_callable_directly(monkeypatch):
    """Unadvertised tools stay reachable by name, never via tools/list."""
    monkeypatch.setattr(server, "cognee_client", FakeStatusClient())
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        assert "cognify_status" not in {tool.name for tool in await client.list_tools()}

        result = await client.call_tool("cognify_status", {"dataset_name": "main_dataset"})
        assert "DATASET_PROCESSING_COMPLETED" in result.content[0].text


async def test_hidden_tool_is_callable_through_the_proxy(monkeypatch):
    """How an agent reaches a tool it just found via search_tools."""
    monkeypatch.setattr(server, "cognee_client", FakeStatusClient())
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "call_tool",
            {"name": "cognify_status", "arguments": {"dataset_name": "main_dataset"}},
        )
        assert "DATASET_PROCESSING_COMPLETED" in result.content[0].text


async def test_proxy_refuses_to_call_the_synthetic_tools():
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        for name in SYNTHETIC_TOOLS:
            with pytest.raises(ToolError):
                await client.call_tool("call_tool", {"name": name, "arguments": {}})


async def test_hidden_tools_keep_their_schemas():
    """Search results must carry enough for an agent to call the tool without a
    second round trip."""
    server.apply_tool_mode("minimal")

    async with Client(server.mcp) as client:
        result = await client.call_tool("search_tools", {"query": "background ingestion status"})
        tools = {tool["name"]: tool for tool in json.loads(result.content[0].text)}

    # Full input schema, so the agent can call it straight away.
    assert "dataset_name" in tools["cognify_status"]["inputSchema"]["properties"]


async def test_usage_logging_name_survives_the_registry_wrapper():
    """@registry.tool folds in @log_usage; every tool must still log as
    'MCP <tool_name>' the way the hand-written decorators did."""
    calls = []

    async def fake_log(**kwargs):
        calls.append(kwargs)

    import cognee.shared.usage_logger as usage_logger

    original_log = usage_logger._log_usage_async
    original_config = usage_logger.get_cache_config

    class Config:
        usage_logging = True

    usage_logger._log_usage_async = fake_log
    usage_logger.get_cache_config = lambda: Config()
    try:
        await server.forget()  # validation-only path, touches no databases
    finally:
        usage_logger._log_usage_async = original_log
        usage_logger.get_cache_config = original_config

    assert [c["function_name"] for c in calls] == ["MCP forget"]
    assert calls[0]["log_type"] == "mcp_tool"
