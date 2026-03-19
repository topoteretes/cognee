"""Tests for cognee.tools — serializers, handler, and definitions."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.tools import (
    remember,
    search_memory,
    handle_tool_call,
    for_openai,
    for_anthropic,
    for_generic,
    TOOLS,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Serializer tests (no cognee runtime needed)
# ---------------------------------------------------------------------------


class TestForOpenai:
    def test_returns_list(self):
        tools = for_openai()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_tool_structure(self):
        tools = for_openai()
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]
            assert "required" in fn["parameters"]

    def test_remember_schema(self):
        tools = for_openai()
        remember_tool = next(t for t in tools if t["function"]["name"] == "remember")
        params = remember_tool["function"]["parameters"]
        assert "content" in params["properties"]
        assert "content" in params["required"]
        assert params["properties"]["content"]["type"] == "string"

    def test_search_memory_schema(self):
        tools = for_openai()
        search_tool = next(t for t in tools if t["function"]["name"] == "search_memory")
        params = search_tool["function"]["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]
        assert "top_k" in params["properties"]
        assert params["properties"]["top_k"]["type"] == "integer"
        # top_k has a default, so it should NOT be required
        assert "top_k" not in params["required"]

    def test_schemas_are_json_serializable(self):
        tools = for_openai()
        json.dumps(tools)


class TestForAnthropic:
    def test_returns_list(self):
        tools = for_anthropic()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_tool_structure(self):
        tools = for_anthropic()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_remember_schema(self):
        tools = for_anthropic()
        remember_tool = next(t for t in tools if t["name"] == "remember")
        schema = remember_tool["input_schema"]
        assert "content" in schema["properties"]
        assert "content" in schema["required"]

    def test_schemas_are_json_serializable(self):
        tools = for_anthropic()
        json.dumps(tools)


class TestForGeneric:
    def test_returns_list(self):
        tools = for_generic()
        assert isinstance(tools, list)
        assert len(tools) == 2

    def test_tool_structure(self):
        tools = for_generic()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_schemas_are_json_serializable(self):
        tools = for_generic()
        json.dumps(tools)


class TestToolNames:
    def test_tool_names_match(self):
        openai_names = {t["function"]["name"] for t in for_openai()}
        anthropic_names = {t["name"] for t in for_anthropic()}
        generic_names = {t["name"] for t in for_generic()}
        assert openai_names == anthropic_names == generic_names == {"remember", "search_memory"}


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestHandleToolCall:
    def test_openai_format(self):
        """OpenAI tool_call: .function.name + .function.arguments (JSON string)."""
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="remember",
                arguments=json.dumps({"content": "test fact"}),
            )
        )
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.add = AsyncMock()
            mock_cognee.cognify = AsyncMock()
            result = _run(handle_tool_call(tool_call))
            assert result == "Remembered."
            mock_cognee.add.assert_awaited_once()

    def test_anthropic_format(self):
        """Anthropic tool_use: .name + .input (dict)."""
        tool_call = SimpleNamespace(
            name="remember",
            input={"content": "test fact"},
        )
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.add = AsyncMock()
            mock_cognee.cognify = AsyncMock()
            result = _run(handle_tool_call(tool_call))
            assert result == "Remembered."

    def test_dict_format(self):
        """Dict-based tool call."""
        tool_call = {
            "name": "remember",
            "arguments": json.dumps({"content": "test fact"}),
        }
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.add = AsyncMock()
            mock_cognee.cognify = AsyncMock()
            result = _run(handle_tool_call(tool_call))
            assert result == "Remembered."

    def test_unknown_tool(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="nonexistent",
                arguments="{}",
            )
        )
        result = _run(handle_tool_call(tool_call))
        assert "Unknown tool" in result

    def test_search_memory_dispatch(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="search_memory",
                arguments=json.dumps({"query": "preferences"}),
            )
        )
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.search = AsyncMock(return_value=["dark mode"])
            result = _run(handle_tool_call(tool_call))
            assert "dark mode" in result


# ---------------------------------------------------------------------------
# Definitions tests
# ---------------------------------------------------------------------------


class TestDefinitions:
    def test_tools_registry(self):
        assert len(TOOLS) == 2
        names = {fn.__name__ for fn in TOOLS}
        assert names == {"remember", "search_memory"}

    def test_remember_calls_add(self):
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.add = AsyncMock()
            mock_cognee.cognify = AsyncMock()
            result = _run(remember("test"))
            assert result == "Remembered."
            mock_cognee.add.assert_awaited_once_with(
                "test", dataset_name="main_dataset"
            )

    def test_search_memory_calls_search(self):
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.search = AsyncMock(return_value=["result"])
            result = _run(search_memory("query"))
            assert "result" in result

    def test_search_memory_handles_empty(self):
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.search = AsyncMock(return_value=[])
            result = _run(search_memory("query"))
            assert "No relevant memories" in result

    def test_search_memory_handles_exception(self):
        with patch("cognee.tools.definitions.cognee") as mock_cognee:
            mock_cognee.search = AsyncMock(side_effect=RuntimeError("db down"))
            result = _run(search_memory("query"))
            assert "failed" in result.lower()
