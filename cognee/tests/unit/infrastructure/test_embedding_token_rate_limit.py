import asyncio
import os
import time

import pytest

import cognee.shared.rate_limiting as rate_limiting
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_config,
)


class _FixedTokenizer:
    """Tokenizer stub: every text costs a fixed number of tokens."""

    def __init__(self, tokens_per_text: int):
        self.tokens_per_text = tokens_per_text
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        return self.tokens_per_text


class _BrokenTokenizer:
    def count_tokens(self, text: str) -> int:
        raise RuntimeError("tokenizer unavailable")


def _configure(enabled: str, requests: str, interval: str, tokens: str) -> None:
    os.environ["EMBEDDING_RATE_LIMIT_ENABLED"] = enabled
    os.environ["EMBEDDING_RATE_LIMIT_REQUESTS"] = requests
    os.environ["EMBEDDING_RATE_LIMIT_INTERVAL"] = interval
    os.environ["EMBEDDING_RATE_LIMIT_TOKENS"] = tokens
    get_embedding_config.cache_clear()
    rate_limiting._embedding_rate_limiter = None
    rate_limiting._embedding_token_rate_limiter = None


@pytest.fixture(autouse=True)
def _restore_environment():
    saved = {
        key: os.environ.get(key)
        for key in (
            "EMBEDDING_RATE_LIMIT_ENABLED",
            "EMBEDDING_RATE_LIMIT_REQUESTS",
            "EMBEDDING_RATE_LIMIT_INTERVAL",
            "EMBEDDING_RATE_LIMIT_TOKENS",
        )
    }
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_embedding_config.cache_clear()
    rate_limiting._embedding_rate_limiter = None
    rate_limiting._embedding_token_rate_limiter = None


def test_count_embedding_tokens_is_zero_when_no_budget_is_configured():
    """The shipped default (EMBEDDING_RATE_LIMIT_TOKENS=0) must not even tokenize."""
    _configure(enabled="true", requests="1000", interval="1", tokens="0")
    tokenizer = _FixedTokenizer(tokens_per_text=10)

    assert rate_limiting.count_embedding_tokens(tokenizer, ["a", "b"]) == 0
    assert tokenizer.calls == 0


def test_count_embedding_tokens_sums_the_batch_when_a_budget_is_set():
    _configure(enabled="true", requests="1000", interval="1", tokens="100")
    tokenizer = _FixedTokenizer(tokens_per_text=10)

    assert rate_limiting.count_embedding_tokens(tokenizer, ["a", "b", "c"]) == 30
    # A bare string is one text, not one text per character.
    assert rate_limiting.count_embedding_tokens(tokenizer, "a") == 10


def test_count_embedding_tokens_survives_a_failing_tokenizer():
    _configure(enabled="true", requests="1000", interval="1", tokens="100")

    assert rate_limiting.count_embedding_tokens(_BrokenTokenizer(), ["a"]) == 0
    assert rate_limiting.count_embedding_tokens(None, ["a"]) == 0


@pytest.mark.asyncio
async def test_token_budget_paces_dispatches_that_exceed_it():
    """Four 30-token dispatches against a 60-token/1s budget must span intervals.

    The request budget is set high enough that it cannot be what paces them.
    """
    _configure(enabled="true", requests="1000", interval="1", tokens="60")
    tokenizer = _FixedTokenizer(tokens_per_text=30)

    async def dispatch():
        async with rate_limiting.embedding_rate_limiter_context_manager():
            await rate_limiting.consume_embedding_token_budget(tokenizer, ["one text"])

    started = time.monotonic()
    await asyncio.gather(*(dispatch() for _ in range(4)))
    elapsed = time.monotonic() - started

    # 120 tokens against 60 tokens/second cannot clear in under a second.
    assert elapsed >= 0.9, f"token budget did not pace the dispatches ({elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_dispatch_larger_than_the_whole_budget_is_admitted():
    """A single oversized dispatch is clamped, not rejected with ValueError."""
    _configure(enabled="true", requests="1000", interval="1", tokens="10")
    tokenizer = _FixedTokenizer(tokens_per_text=10_000)

    async with rate_limiting.embedding_rate_limiter_context_manager():
        await rate_limiting.consume_embedding_token_budget(tokenizer, ["huge"])


@pytest.mark.asyncio
async def test_no_token_budget_leaves_dispatch_unpaced():
    """With the default token budget, only the request limiter applies."""
    _configure(enabled="true", requests="1000", interval="1", tokens="0")
    tokenizer = _FixedTokenizer(tokens_per_text=10_000)

    async def dispatch():
        async with rate_limiting.embedding_rate_limiter_context_manager():
            await rate_limiting.consume_embedding_token_budget(tokenizer, ["one text"])

    started = time.monotonic()
    await asyncio.gather(*(dispatch() for _ in range(8)))
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"unbudgeted dispatches were paced ({elapsed:.2f}s)"


def test_token_budget_is_serialized_with_its_siblings():
    _configure(enabled="true", requests="60", interval="60", tokens="4242")

    assert get_embedding_config().to_dict()["embedding_rate_limit_tokens"] == 4242
