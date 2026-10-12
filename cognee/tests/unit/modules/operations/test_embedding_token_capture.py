"""Embedding requests feed the operation accumulator like LLM completions.

Before this, no embedding engine reported usage, so an operation that only
embeds (``recall``, a session ``remember``) measured zero tokens and the
activity feed hid it as "nothing to show". No network: the engines' shared
``record_embedding_usage`` helper is driven with synthetic responses.
"""

from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.vector.embeddings.utils import record_embedding_usage
from cognee.modules.operations.usage_accumulator import operation_usage_scope
from cognee.modules.session_lifecycle.usage_tracking import (
    estimate_cost_usd,
    record_embedding_call,
)


@pytest.mark.asyncio
async def test_embedding_tokens_reach_the_operation_accumulator():
    with operation_usage_scope() as usage:
        await record_embedding_call(model="text-embedding-3-large", tokens_in=37)

    assert usage.tokens_in == 37
    # Embeddings have no completion side: a measured zero, not an estimate.
    assert usage.tokens_out == 0


@pytest.mark.asyncio
async def test_litellm_style_usage_object_is_read():
    response = SimpleNamespace(
        data=[{"embedding": [0.0]}],
        usage=SimpleNamespace(prompt_tokens=12, total_tokens=12),
    )
    with operation_usage_scope() as usage:
        await record_embedding_usage(response, "litellm_proxy/text-embedding-3-large")

    assert (usage.tokens_in, usage.tokens_out) == (12, 0)


@pytest.mark.asyncio
async def test_dict_usage_falls_back_to_total_tokens():
    with operation_usage_scope() as usage:
        await record_embedding_usage({"data": [], "usage": {"total_tokens": 5}}, "m")

    assert usage.tokens_in == 5


@pytest.mark.asyncio
async def test_missing_usage_records_nothing_and_never_raises():
    with operation_usage_scope() as usage:
        await record_embedding_usage(SimpleNamespace(data=[]), "m")
        await record_embedding_usage(None, "m")
        await record_embedding_usage(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None)), "m"
        )

    assert (usage.tokens_in, usage.tokens_out) == (0, 0)


@pytest.mark.asyncio
async def test_nested_scopes_chain_embedding_tokens_to_the_parent():
    with operation_usage_scope() as outer, operation_usage_scope() as inner:
        await record_embedding_call(model="text-embedding-3-small", tokens_in=4)
    assert inner.tokens_in == 4
    assert outer.tokens_in == 4


def test_embedding_models_are_priced_on_the_input_side_only():
    # Provider prefixes are stripped the same way as for chat models.
    assert estimate_cost_usd("litellm_proxy/text-embedding-3-large", 1_000_000, 0) == pytest.approx(
        0.13
    )
    assert estimate_cost_usd("text-embedding-3-small", 1_000_000, 0) == pytest.approx(0.02)
