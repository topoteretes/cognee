"""extract_summary keeps its single-value contract; extract_summary_with_provenance
(SDK-529) returns the prompt text and the stage model alongside the LLM output.

The gateway call is patched — no LLM, no network.
"""

import importlib
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from cognee.context_global_variables import llm_config as llm_config_ctx
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage

extract_summary_module = importlib.import_module(
    "cognee.infrastructure.llm.extraction.extract_summary"
)

PROMPT = "PROMPT<summarize_content.txt>"


class Summary(BaseModel):
    summary: str


@pytest.fixture
def gateway(monkeypatch):
    """Patch the gateway and the prompt read; returns (mock, llm_output)."""
    output = Summary(summary="short")
    mock = AsyncMock(return_value=output)
    monkeypatch.setattr(extract_summary_module.LLMGateway, "acreate_structured_output", mock)
    monkeypatch.setattr(extract_summary_module, "read_query_prompt", lambda name: f"PROMPT<{name}>")
    return mock, output


@pytest.fixture
def llm_context():
    """Bind an LLMConfig with a summarization-stage override on the LLM ContextVar."""
    config = LLMConfig(
        llm_provider="openai",
        llm_model="base-model",
        llm_endpoint="",
        llm_api_key="base-key",
        llm_summarization_model="summarization-model",
    )
    token = llm_config_ctx.set(config)
    try:
        yield config
    finally:
        llm_config_ctx.reset(token)


@pytest.mark.asyncio
async def test_extract_summary_with_provenance_returns_output_prompt_and_model(
    gateway, llm_context
):
    mock, output = gateway

    result = await extract_summary_module.extract_summary_with_provenance("content", Summary)

    assert result == (output, PROMPT, "base-model")
    mock.assert_awaited_once_with("content", PROMPT, Summary)


@pytest.mark.asyncio
async def test_extract_summary_with_provenance_reports_the_stage_model(gateway, llm_context):
    _mock, output = gateway

    with pipeline_stage("summarization"):
        result = await extract_summary_module.extract_summary_with_provenance("content", Summary)

    assert result == (output, PROMPT, "summarization-model")


@pytest.mark.asyncio
async def test_extract_summary_keeps_its_single_value_contract(gateway, llm_context):
    mock, output = gateway

    assert await extract_summary_module.extract_summary("content", Summary) is output
    mock.assert_awaited_once_with("content", PROMPT, Summary)


@pytest.mark.asyncio
async def test_extract_summary_with_provenance_tolerates_a_missing_prompt_file(
    gateway, llm_context, monkeypatch
):
    mock, output = gateway
    monkeypatch.setattr(extract_summary_module, "read_query_prompt", lambda name: None)

    result = await extract_summary_module.extract_summary_with_provenance("content", Summary)

    # Same fallback as before: an unreadable prompt file sends an empty system prompt.
    assert result == (output, "", "base-model")
    mock.assert_awaited_once_with("content", "", Summary)
