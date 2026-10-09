"""Token hook tests: ``record_llm_call`` feeds the operation accumulator (SDK-399).

``record_llm_call`` is the single choke point for counted LLM calls. The
operation accumulator is session-independent: tokens are captured with NO
active session scope (the core of requirement f — previously those tokens
were dropped), and when both scopes are active the same counts feed both
sinks, so operation rows and ``SessionModelUsage`` reconcile.

No LLM, no network: the hook is driven directly with synthetic texts.
"""

import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.operations.usage_accumulator import operation_usage_scope

usage_tracking = importlib.import_module("cognee.modules.session_lifecycle.usage_tracking")

# 40 chars -> 10 tokens, 20 chars -> 5 tokens under the chars/4 estimate.
INPUT_TEXT = "a" * 40
OUTPUT_TEXT = "b" * 20


@pytest.mark.asyncio
async def test_tokens_captured_outside_any_session_scope():
    """A plain cognify() (no session) still measures its token spend."""
    with operation_usage_scope() as usage:
        await usage_tracking.record_llm_call(input_text=INPUT_TEXT, output_text=OUTPUT_TEXT)

    assert usage.tokens_in == usage_tracking._estimate_tokens(INPUT_TEXT) == 10
    assert usage.tokens_out == usage_tracking._estimate_tokens(OUTPUT_TEXT) == 5


@pytest.mark.asyncio
async def test_token_overrides_reach_the_accumulator():
    with operation_usage_scope() as usage:
        await usage_tracking.record_llm_call(
            input_text=INPUT_TEXT,
            output_text=OUTPUT_TEXT,
            tokens_in_override=123,
            tokens_out_override=45,
        )

    assert (usage.tokens_in, usage.tokens_out) == (123, 45)


@pytest.mark.asyncio
async def test_no_scope_and_no_session_is_a_no_op():
    """Without any active scope the hook must not raise or persist anything."""
    await usage_tracking.record_llm_call(input_text=INPUT_TEXT, output_text=OUTPUT_TEXT)


@pytest.mark.asyncio
async def test_session_and_operation_scopes_receive_identical_counts(monkeypatch):
    """Reconciliation: one call feeds both sinks with the same numbers."""
    accumulate_usage = AsyncMock()
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.accumulate_usage", accumulate_usage
    )

    session_id = "session-1"
    user_id = uuid4()

    with operation_usage_scope() as usage:
        async with usage_tracking.track_session_usage(session_id, user_id):
            await usage_tracking.record_llm_call(
                input_text=INPUT_TEXT, output_text=OUTPUT_TEXT, model="gpt-5-mini"
            )

    assert (usage.tokens_in, usage.tokens_out) == (10, 5)

    accumulate_usage.assert_awaited_once()
    session_kwargs = accumulate_usage.await_args.kwargs
    assert session_kwargs["session_id"] == session_id
    assert session_kwargs["user_id"] == user_id
    assert session_kwargs["tokens_in"] == usage.tokens_in
    assert session_kwargs["tokens_out"] == usage.tokens_out


@pytest.mark.asyncio
async def test_session_only_path_unchanged_without_operation_scope(monkeypatch):
    """The pre-existing session grain keeps working with no operation active."""
    accumulate_usage = AsyncMock()
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.accumulate_usage", accumulate_usage
    )

    async with usage_tracking.track_session_usage("session-2", uuid4()):
        await usage_tracking.record_llm_call(input_text=INPUT_TEXT, output_text=OUTPUT_TEXT)

    accumulate_usage.assert_awaited_once()
    assert accumulate_usage.await_args.kwargs["tokens_in"] == 10
    assert accumulate_usage.await_args.kwargs["tokens_out"] == 5
