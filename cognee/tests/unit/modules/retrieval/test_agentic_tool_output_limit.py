from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.retrieval import agentic_retriever
from cognee.modules.retrieval.agentic_retriever import AgenticRetriever, AgentStep, ToolCall

TOOL_OUTPUT = "x" * (agentic_retriever.MAX_TOOL_OUTPUT_CHARS + 500)


async def _context_after_one_tool_call(**retriever_kwargs) -> str:
    """Run one tool call that returns TOOL_OUTPUT; return the context the agent reads next."""
    retriever = AgenticRetriever(
        user=SimpleNamespace(id=None), dataset_id=uuid4(), **retriever_kwargs
    )
    steps = [
        AgentStep(tool_call=ToolCall(tool_name="read_table", arguments={})),
        AgentStep(final_answer="done"),
    ]
    with (
        patch.object(agentic_retriever, "generate_completion", AsyncMock(side_effect=steps)) as llm,
        patch.object(retriever, "_run_tool_safely", AsyncMock(return_value=TOOL_OUTPUT)),
        patch.object(retriever, "_maybe_active_context_block", AsyncMock(return_value=("", []))),
        patch.object(retriever, "_get_session_history", AsyncMock(return_value="")),
    ):
        assert await retriever._run_tool_loop("q", "", ["read_table"]) == "done"
    return llm.await_args_list[1].kwargs["context"]


@pytest.mark.asyncio
async def test_tool_output_is_cut_at_the_default_limit():
    context = await _context_after_one_tool_call()

    assert context.count("x") == agentic_retriever.MAX_TOOL_OUTPUT_CHARS
    assert context.endswith("... [truncated]")


@pytest.mark.asyncio
async def test_max_tool_output_chars_raises_the_limit():
    context = await _context_after_one_tool_call(max_tool_output_chars=len(TOOL_OUTPUT))

    assert context.endswith(TOOL_OUTPUT)
