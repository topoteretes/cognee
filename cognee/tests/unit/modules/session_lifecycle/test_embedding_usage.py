"""Embedding usage belongs in the same totals as server-side generation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.session_lifecycle import usage_tracking as tracking


@pytest.mark.asyncio
async def test_embedding_usage_preserves_provider_tokens_and_cost(monkeypatch):
    record = AsyncMock()
    monkeypatch.setattr(tracking, "record_llm_call", record)
    monkeypatch.setattr("litellm.completion_cost", lambda **kw: 0.001)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=37))
    await tracking.record_embedding_call(response, model="embedding-model", input_text="text")
    assert record.call_args.kwargs == {
        "input_text": "text",
        "output_text": "",
        "model": "embedding-model",
        "tokens_in_override": 37,
        "tokens_out_override": 0,
        "cost_usd_override": 0.001,
    }


@pytest.mark.asyncio
async def test_unpriced_embedding_retains_tokens_and_cannot_break_capture(monkeypatch):
    record = AsyncMock(side_effect=OSError("metrics unavailable"))
    monkeypatch.setattr(tracking, "record_llm_call", record)

    def unknown_price(**kw):
        raise ValueError("unknown model")

    monkeypatch.setattr("litellm.completion_cost", unknown_price)
    await tracking.record_embedding_call(
        SimpleNamespace(usage={"prompt_tokens": 12}), model="custom", input_text="text"
    )
    assert record.call_args.kwargs["tokens_in_override"] == 12
    assert record.call_args.kwargs["cost_usd_override"] is None
