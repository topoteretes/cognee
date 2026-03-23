"""Tests for cognee.tools — serializers, handler, and definitions."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cognee.tools import (
    remember,
    recall,
    handle_tool_call,
    for_openai,
    for_anthropic,
    for_generic,
    TOOLS,
)


def _run(coro):
    """Run an async coroutine synchronously in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


class TestSerializers:
    def test_openai_format(self):
        tools = for_openai()
        assert len(tools) == 2
        for tool in tools:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]
            assert "required" in fn["parameters"]

    def test_anthropic_format(self):
        tools = for_anthropic()
        assert len(tools) == 2
        for tool in tools:
            assert "name" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_generic_format(self):
        tools = for_generic()
        assert len(tools) == 2
        for tool in tools:
            assert "name" in tool
            assert "parameters" in tool

    def test_names_consistent_across_formats(self):
        openai_names = {t["function"]["name"] for t in for_openai()}
        anthropic_names = {t["name"] for t in for_anthropic()}
        generic_names = {t["name"] for t in for_generic()}
        assert openai_names == anthropic_names == generic_names == {"remember", "recall"}

    def test_remember_has_content_required(self):
        tool = next(t for t in for_openai() if t["function"]["name"] == "remember")
        params = tool["function"]["parameters"]
        assert "content" in params["required"]

    def test_recall_has_query_required_topk_optional(self):
        tool = next(t for t in for_openai() if t["function"]["name"] == "recall")
        params = tool["function"]["parameters"]
        assert "query" in params["required"]
        assert "top_k" not in params["required"]

    def test_json_serializable(self):
        json.dumps(for_openai())
        json.dumps(for_anthropic())
        json.dumps(for_generic())

    def test_langchain_import_error(self):
        """for_langchain() raises ImportError when langchain-core is missing."""
        import pytest
        from cognee.tools.serializers.langchain import for_langchain

        with pytest.raises(ImportError, match="langchain-core"):
            for_langchain()

    def test_crewai_import_error(self):
        """for_crewai() raises ImportError when crewai is missing."""
        import pytest
        from cognee.tools.serializers.crewai import for_crewai

        with pytest.raises(ImportError, match="crewai"):
            for_crewai()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class TestHandleToolCall:
    def test_openai_format(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="remember",
                arguments=json.dumps({"content": "test fact"}),
            )
        )
        with patch("cognee.tools.definitions.v2_remember", new_callable=AsyncMock):
            result = _run(handle_tool_call(tool_call))
            assert result == "Remembered."

    def test_anthropic_format(self):
        tool_call = SimpleNamespace(
            name="remember",
            input={"content": "test fact"},
        )
        with patch("cognee.tools.definitions.v2_remember", new_callable=AsyncMock):
            result = _run(handle_tool_call(tool_call))
            assert result == "Remembered."

    def test_dict_format(self):
        tool_call = {
            "name": "recall",
            "arguments": json.dumps({"query": "preferences"}),
        }
        with patch(
            "cognee.tools.definitions.v2_recall",
            new_callable=AsyncMock,
            return_value=["dark mode"],
        ):
            result = _run(handle_tool_call(tool_call))
            assert "dark mode" in result

    def test_unknown_tool(self):
        tool_call = SimpleNamespace(function=SimpleNamespace(name="nonexistent", arguments="{}"))
        result = _run(handle_tool_call(tool_call))
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


class TestDefinitions:
    def test_tools_registry(self):
        assert len(TOOLS) == 2
        assert {fn.__name__ for fn in TOOLS} == {"remember", "recall"}

    def test_remember_calls_v2(self):
        with patch("cognee.tools.definitions.v2_remember", new_callable=AsyncMock) as mock:
            result = _run(remember("test"))
            assert result == "Remembered."
            mock.assert_awaited_once_with(data="test", dataset_name="main_dataset")

    def test_recall_returns_result(self):
        with patch(
            "cognee.tools.definitions.v2_recall",
            new_callable=AsyncMock,
            return_value=["answer"],
        ):
            assert "answer" in _run(recall("query"))

    def test_recall_handles_empty(self):
        with patch(
            "cognee.tools.definitions.v2_recall",
            new_callable=AsyncMock,
            return_value=[],
        ):
            assert "No relevant memories" in _run(recall("query"))

    def test_recall_handles_exception(self):
        with patch(
            "cognee.tools.definitions.v2_recall",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        ):
            assert "failed" in _run(recall("query")).lower()
