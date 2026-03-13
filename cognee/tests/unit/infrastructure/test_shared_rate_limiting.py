"""Tests for cognee.shared.rate_limiting"""

import asyncio
import time
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: reset cached limiters between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_limiters():
    """Ensure every test starts with a clean limiter cache."""
    from cognee.shared.rate_limiting import reset_rate_limiters

    reset_rate_limiters()
    yield
    reset_rate_limiters()


# ---------------------------------------------------------------------------
# A. estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic():
    from cognee.shared.rate_limiting import estimate_tokens

    count = estimate_tokens("hello world")
    # tiktoken cl100k_base encodes "hello world" as 2 tokens
    assert count >= 1
    assert isinstance(count, int)


def test_estimate_tokens_empty():
    from cognee.shared.rate_limiting import estimate_tokens

    assert estimate_tokens("") == 1  # minimum is 1


def test_estimate_tokens_multiple_strings():
    from cognee.shared.rate_limiting import estimate_tokens

    combined = estimate_tokens("hello", "world")
    separate = estimate_tokens("hello") + estimate_tokens("world")
    assert combined == separate


def test_estimate_tokens_varargs_no_concatenation():
    """Calling with varargs should NOT allocate a concatenated string."""
    from cognee.shared.rate_limiting import estimate_tokens

    a = "hello " * 100
    b = "world " * 100
    # Just verify it works and returns an int — the point is the API accepts *args
    result = estimate_tokens(a, b)
    assert result > 10


# ---------------------------------------------------------------------------
# B. Limiter behaviour
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a mock LLMConfig with sensible defaults, applying overrides."""
    defaults = {
        "llm_rate_limit_enabled": False,
        "llm_rate_limit_requests": 60,
        "llm_rate_limit_interval": 60,
        "llm_rate_limit_tokens": 0,
        "embedding_rate_limit_enabled": False,
        "embedding_rate_limit_requests": 60,
        "embedding_rate_limit_interval": 60,
        "embedding_rate_limit_tokens": 0,
    }
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.asyncio
async def test_disabled_returns_nullcontext():
    cfg = _make_config(llm_rate_limit_enabled=False)
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg):
        from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

        ctx = llm_rate_limiter_context_manager(tokens=100)
        assert isinstance(ctx, type(nullcontext()))


@pytest.mark.asyncio
async def test_enabled_throttles_requests():
    """With a tiny limit (2 req / 10s), 4 concurrent calls should take measurably longer."""
    cfg = _make_config(
        llm_rate_limit_enabled=True,
        llm_rate_limit_requests=2,
        llm_rate_limit_interval=10,
    )
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg):
        from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

        start = time.monotonic()

        async def _acquire():
            async with llm_rate_limiter_context_manager(tokens=0):
                pass

        # Fire 4 requests; only 2 can go through immediately
        await asyncio.gather(*[_acquire() for _ in range(4)])
        elapsed = time.monotonic() - start
        # The 3rd+4th requests must wait, so elapsed should be > 0
        # (aiolimiter will block them until capacity refills)
        assert elapsed > 0.01


@pytest.mark.asyncio
async def test_token_cap_no_valueerror():
    """Requesting more tokens than max_tokens should NOT raise ValueError."""
    cfg = _make_config(
        llm_rate_limit_enabled=True,
        llm_rate_limit_tokens=100,
    )
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg):
        from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

        # tokens=99999 > max_tokens=100, should be capped, not error
        async with llm_rate_limiter_context_manager(tokens=99999):
            pass


@pytest.mark.asyncio
async def test_request_and_token_limiter_both_active():
    """Both request and token gates should be honoured."""
    cfg = _make_config(
        llm_rate_limit_enabled=True,
        llm_rate_limit_requests=100,
        llm_rate_limit_tokens=1000,
    )
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg):
        from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

        # Should not raise
        async with llm_rate_limiter_context_manager(tokens=500):
            pass


# ---------------------------------------------------------------------------
# C. Lazy init + config reload
# ---------------------------------------------------------------------------


def test_lazy_init():
    """Importing the module should NOT create any limiters."""
    import cognee.shared.rate_limiting as rl

    assert rl._llm_request_limiter is None
    assert rl._embedding_request_limiter is None


def test_reset_rate_limiters():
    """After first use, reset should clear cached limiters."""
    cfg = _make_config(llm_rate_limit_enabled=True)
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg):
        import cognee.shared.rate_limiting as rl

        # Trigger lazy init
        rl.llm_rate_limiter_context_manager(tokens=0)
        assert rl._llm_request_limiter is not None

        rl.reset_rate_limiters()
        assert rl._llm_request_limiter is None


def test_config_change_after_reset():
    """After reset, new config values should be picked up."""
    import cognee.shared.rate_limiting as rl

    cfg1 = _make_config(llm_rate_limit_enabled=True, llm_rate_limit_requests=10)
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg1):
        rl.llm_rate_limiter_context_manager(tokens=0)
        limiter1 = rl._llm_request_limiter
        assert limiter1 is not None

    rl.reset_rate_limiters()

    cfg2 = _make_config(llm_rate_limit_enabled=True, llm_rate_limit_requests=99)
    with patch("cognee.shared.rate_limiting.get_llm_config", return_value=cfg2):
        rl.llm_rate_limiter_context_manager(tokens=0)
        limiter2 = rl._llm_request_limiter
        assert limiter2 is not None
        # Should be a different limiter instance
        assert limiter1 is not limiter2
