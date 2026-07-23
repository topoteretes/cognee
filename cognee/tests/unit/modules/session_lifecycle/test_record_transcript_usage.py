"""Tests for transcript-derived token approximation.

Agent-facing operations (remember / @agent_memory tool calls) never enter the
LLM-completion usage scope, so their session rows would show 0 tokens (and thus
$0 cost). ``record_transcript_usage`` approximates tokens from the trace step's
text and accumulates them onto the session. These tests pin that behavior.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.session_lifecycle.usage_tracking import record_transcript_usage

ACCUMULATE = "cognee.modules.session_lifecycle.metrics.accumulate_usage"


@pytest.mark.asyncio
async def test_estimates_tokens_and_accumulates_tokens_only():
    uid = uuid4()
    with patch(ACCUMULATE, new=AsyncMock()) as acc:
        # ~chars/4 heuristic: 400 -> 100 in, 80 -> 20 out.
        await record_transcript_usage(
            session_id="s1", user_id=str(uid), input_text="x" * 400, output_text="y" * 80
        )
    acc.assert_awaited_once()
    kwargs = acc.await_args.kwargs
    assert kwargs["tokens_in"] == 100
    assert kwargs["tokens_out"] == 20
    assert kwargs["session_id"] == "s1"
    assert kwargs["user_id"] == uid  # coerced from str to UUID
    # Cost is priced downstream from tokens, so none is written here.
    assert "cost_usd" not in kwargs


@pytest.mark.asyncio
async def test_no_accumulate_when_no_text():
    with patch(ACCUMULATE, new=AsyncMock()) as acc:
        await record_transcript_usage(session_id="s2", user_id=str(uuid4()))
    acc.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_user_id_is_swallowed():
    with patch(ACCUMULATE, new=AsyncMock()) as acc:
        await record_transcript_usage(
            session_id="s3", user_id="not-a-uuid", input_text="hello world"
        )
    acc.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepts_uuid_directly():
    uid = uuid4()
    with patch(ACCUMULATE, new=AsyncMock()) as acc:
        await record_transcript_usage(session_id="s4", user_id=uid, input_text="a" * 40)
    acc.assert_awaited_once()
    assert acc.await_args.kwargs["user_id"] == uid
