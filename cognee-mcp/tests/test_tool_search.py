"""Behavior of the tool-search gating added in CLO-352.

The guarantee these tests protect: shrinking what ``tools/list`` advertises must
not shrink what is *callable*. Every unadvertised tool has to stay reachable
both directly by name (how the workspace UI calls its internals) and through
the ``call_tool`` proxy (how an agent calls something it just found).
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
WORKSPACE_UI_ENTRY_TOOLS = {"visualize_graph_ui", "upload_file_ui", "open_cognee_workspace"}
WORKSPACE_INTERNAL_TOOLS = {
    "cognify_file",
    "list_datasets_json",
    "list_dataset_data_json",
    "create_dataset_json",
    "get_client_info_json",
}


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


def test_every_tool_declares_a_tier():
    """A tool registered without going through @registry.tool would be missed by
    names_with_tag(), so assert the registry covers the whole catalog."""
    assert set(server.registry.tags) == (
        MEMORY_TOOLS | WORKSPACE_UI_ENTRY_TOOLS | WORKSPACE_INTERNAL_TOOLS
    )
    assert all(tags for tags in server.registry.tags.values())


def test_pinned_sets_are_derived_from_tags():
    assert set(server.registry.names_with_tag(MEMORY_TAG)) == MEMORY_TOOLS
    assert set(server.registry.names_with_tag(DEFAULT_TAG)) == (
        MEMORY_TOOLS | WORKSPACE_UI_ENTRY_TOOLS
    )


# --- what each mode advertises ------------------------------------------------


async def test_default_mode_advertises_pinned_plus_synthetic():
    assert await advertised("default") == (
        MEMORY_TOOLS | WORKSPACE_UI_ENTRY_TOOLS | SYNTHETIC_TOOLS
    )


async def test_minimal_mode_advertises_only_the_memory_api():
    assert await advertised("minimal") == MEMORY_TOOLS | SYNTHETIC_TOOLS


async def test_all_mode_restores_the_flat_surface():
    names = await advertised("all")
    assert names == set(server.registry.tags)
    assert not names & SYNTHETIC_TOOLS


async def test_unknown_mode_falls_back_to_default():
    assert server.apply_tool_mode("banana") == "default"
    assert {tool.name for tool in await server.mcp.list_tools()} == (
        MEMORY_TOOLS | WORKSPACE_UI_ENTRY_TOOLS | SYNTHETIC_TOOLS
    )


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
    hidden = set(server.registry.tags) - set(server.registry.names_with_tag(DEFAULT_TAG))
    assert len(hidden) <= server.TOOL_SEARCH_MAX_RESULTS


async def test_search_matches_on_vocabulary_not_on_the_result_limit():
    """The window being wide enough does NOT mean every search returns everything.

    BM25 drops zero-scoring tools and its tokenizer does no stemming, so "dataset"
    does not match `list_datasets_json` (token "datasets"). This is the behavior
    that makes tool *descriptions* the lever for recall, and it is surprising
    enough to pin: if a future fastmcp adds stemming, we want to notice.
    """
    server.apply_tool_mode("default")
    hidden = set(server.registry.tags) - set(server.registry.names_with_tag(DEFAULT_TAG))

    async with Client(server.mcp) as client:
        singular = set(await search(client, "dataset"))
        plural = set(await search(client, "datasets"))
        combined = set(await search(client, "dataset datasets file client info create list"))

    assert "create_dataset_json" in singular
    assert "list_datasets_json" not in singular, "unexpected stemming — revisit the docs"
    assert "list_datasets_json" in plural
    # Covering both forms does reach everything, confirming the limit is not the cap.
    assert combined == hidden


async def test_hidden_tools_are_discoverable_by_search():
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        assert "list_datasets_json" in await search(client, "list all of my datasets")
        assert "cognify_file" in await search(client, "ingest an uploaded file")


async def test_search_never_returns_pinned_tools():
    """Pinned tools are already in tools/list; echoing them back would waste the
    result budget the search transform exists to protect."""
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        for query in ("remember this for later", "search my memory", "delete a dataset"):
            assert not (MEMORY_TOOLS | WORKSPACE_UI_ENTRY_TOOLS) & set(await search(client, query))


async def test_hidden_tool_is_callable_directly(monkeypatch):
    """How the workspace UI reaches its internals: by name, never via tools/list."""

    class FakeClient:
        async def list_datasets(self):
            return [{"id": "id-1", "name": "alpha"}]

    monkeypatch.setattr(server, "cognee_client", FakeClient())
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        assert "list_datasets_json" not in {tool.name for tool in await client.list_tools()}

        result = await client.call_tool("list_datasets_json", {})
        assert "alpha (id-1)" in result.content[0].text


async def test_hidden_tool_is_callable_through_the_proxy(monkeypatch):
    """How an agent reaches a tool it just found via search_tools."""

    class FakeClient:
        async def list_datasets(self):
            return [{"id": "id-2", "name": "beta"}]

    monkeypatch.setattr(server, "cognee_client", FakeClient())
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "call_tool", {"name": "list_datasets_json", "arguments": {}}
        )
        assert "beta (id-2)" in result.content[0].text


async def test_proxy_refuses_to_call_the_synthetic_tools():
    server.apply_tool_mode("default")

    async with Client(server.mcp) as client:
        for name in SYNTHETIC_TOOLS:
            with pytest.raises(ToolError):
                await client.call_tool("call_tool", {"name": name, "arguments": {}})


async def test_hidden_tools_keep_their_schemas_and_ui_metadata():
    """Search results must carry enough for an agent to call the tool without a
    second round trip, and the workspace UI's resourceUri must survive."""
    server.apply_tool_mode("minimal")

    async with Client(server.mcp) as client:
        result = await client.call_tool("search_tools", {"query": "data items inside a dataset"})
        tools = {tool["name"]: tool for tool in json.loads(result.content[0].text)}

    # Full input schema, so the agent can call it straight away.
    assert "dataset_id" in tools["list_dataset_data_json"]["inputSchema"]["properties"]
    # MCP App metadata survives the transform.
    assert tools["visualize_graph_ui"]["meta"]["ui"]["resourceUri"]


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
