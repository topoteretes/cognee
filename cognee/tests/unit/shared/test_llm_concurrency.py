"""Unit tests for the LLM in-flight concurrency cap (per backend model).

Pure-asyncio tests (no backend): helper tasks hold a "request" open on a
time-based gate (opens on demand, auto-closes after a short window) and
release only when the test says so. Assertions are on OBSERVED concurrency —
how many requests were in flight simultaneously — so the tests prove the cap
holds instead of merely proving a semaphore object was constructed. The
"uncapped" tests prove disabled caps are non-blocking: they complete even
with more tasks open than any plausible cap.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import pytest

import cognee.shared.rate_limiting as rl
from cognee.infrastructure.llm.config import (
    LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS,
    LLMConfig,
)

pytestmark = pytest.mark.asyncio


# --- Fixture: pristine seam state ----------------------------------------------


@pytest.fixture(autouse=True)
def _clean_seam_state():
    """Each test gets fresh limiters/semaphores/policy; nothing leaks."""
    rl._llm_rate_limiter = None
    rl._embedding_rate_limiter = None
    rl._concurrency_state.clear()
    from cognee.infrastructure.llm.overload_policy import llm_overload_policy

    llm_overload_policy._paced_until = 0.0
    yield
    rl._llm_rate_limiter = None
    rl._embedding_rate_limiter = None
    rl._concurrency_state.clear()
    llm_overload_policy._paced_until = 0.0


# --- Helpers ------------------------------------------------------------------


class _Gate:
    """Time-based open/closed gate.

    ``open_for(seconds)`` lets acquirers through for that window and then
    closes again — so a forgotten open can never hang the suite. ``wait()``
    returns once the gate has been open long enough to be considered in
    flight. A closed gate is re-checked in a loop because asyncio gives
    cancelled waiters no wake-up ordering guarantees.
    """

    def __init__(self):
        self._open_until = 0.0

    def open_for(self, seconds):
        self._open_until = max(self._open_until, time.monotonic() + seconds)

    async def wait(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() >= self._open_until:
            if time.monotonic() > deadline:
                raise TimeoutError("_Gate did not open within its timeout")
            await asyncio.sleep(0.01)


class _Held:
    """A dispatch task parked inside the seam, holding its slot + permit."""

    def __init__(self, task, in_flight, handle):
        self._task = task
        self._in_flight = in_flight
        self._handle = handle

    async def entered(self, timeout=10.0):
        """Wait until this request is observably in flight."""
        deadline = time.monotonic() + timeout
        while self._in_flight.value == 0:
            if self._task.done():
                # Surface seam errors immediately instead of timing out.
                exc = self._task.exception()
                if exc is not None:
                    raise exc
                raise RuntimeError("dispatch task ended before entering seam")
            if time.monotonic() > deadline:
                raise TimeoutError("dispatch never reached the seam")
            await asyncio.sleep(0.01)

    async def release(self):
        """Let the dispatch return and wait for the task to finish."""
        self._handle.cancel()
        await self._task


def _hold_open(seam_factory, gate, in_flight, config=None):
    """Start a task that enters ``seam_factory()()`` and parks until released.

    A *factory* (e.g. ``_seam``) is passed, not a context-manager instance:
    each concurrent request needs its own seam object. The gate lives INSIDE
    the context: entering the seam (semaphore acquire + RPM token) happens
    first, then the task waits for "backend ready" while holding its slot —
    like a real dispatch awaiting a slow response. ``config``, when given, is
    bound on the task's own context so the seam reads it (tasks do not
    inherit the caller's ContextVar state).
    """

    async def _request():
        token = None
        if config is not None:
            from cognee.context_global_variables import llm_config as llm_config_ctx

            token = llm_config_ctx.set(config)
        try:
            try:
                async with seam_factory():
                    await gate.wait()
                    in_flight.increment()
                    try:
                        # Parked "in flight": resolved by release() or the test gate.
                        fut = asyncio.get_running_loop().create_future()
                        try:
                            await fut
                        except asyncio.CancelledError:
                            pass
                    finally:
                        in_flight.decrement()
            except asyncio.CancelledError:
                # Normal in this harness: release() cancels a request that is
                # still queued on the seam's semaphore (never entered it).
                pass
        finally:
            if token is not None:
                from cognee.context_global_variables import llm_config as llm_config_ctx

                llm_config_ctx.reset(token)

    task = asyncio.ensure_future(_request())
    return _Held(task, in_flight, task)


class _Counter:
    """Atomic-ish in-flight counter with a high-water mark."""

    def __init__(self):
        self.value = 0
        self.max_seen = 0

    def increment(self):
        self.value += 1
        self.max_seen = max(self.max_seen, self.value)

    def decrement(self):
        self.value -= 1


@asynccontextmanager
async def _seam():
    """The production seam exactly as adapters call it."""
    async with rl.llm_rate_limiter_context_manager():
        yield


def _set_env(monkeypatch, **values):
    # Always clear the credential triple first: cognee loads .env into
    # os.environ at import, and pydantic-settings treats "some but not all"
    # of (LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY) as a validation error. Same
    # isolation trick as test_llm_config_rate_limit_defaults.py.
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for key, value in values.items():
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        if value is not None:
            monkeypatch.setenv(key, str(value))
    from cognee.infrastructure.llm import config as llm_config_module

    llm_config_module.get_llm_config.cache_clear()


def _build(**kwargs):
    """Build an LLMConfig from kwargs alone, never the ambient env."""
    defaults = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": "test-key",
        "llm_endpoint": "https://api.openai.com",
        "_env_file": None,
    }
    defaults.update(kwargs)
    return LLMConfig(**defaults)


# --- Tests --------------------------------------------------------------------


async def test_default_is_unlimited(monkeypatch):
    """With no env set, the config cap is 0 (disabled) — backward compatible."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=None,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    config = _build()
    assert config.llm_max_concurrent_requests == 0


async def test_cloud_default_rpm_is_60(monkeypatch):
    """Cloud providers keep the 60 RPM default when nothing is set."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=None,
        LLM_RATE_LIMIT_REQUESTS=None,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    config = _build()
    assert config.llm_rate_limit_requests == 60


async def test_local_provider_gets_concurrency_default(monkeypatch):
    """Ollama/llama.cpp/LM Studio get the local in-flight cap by default."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=None,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    for provider, model in (
        ("ollama", "phi4:latest"),
        ("llama_cpp", "some-model"),
        ("custom", "lm_studio/qwen2.5-7b"),
    ):
        config = _build(llm_provider=provider, llm_model=model)
        assert config.llm_max_concurrent_requests == LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS


async def test_local_default_value_is_5(monkeypatch):
    """The local default cap is 5 (maintainer-requested value)."""
    assert LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS == 5


async def test_explicit_cap_overrides_local_default(monkeypatch):
    """An explicit LLM_MAX_CONCURRENT_REQUESTS wins over the local default."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=8,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    config = _build(llm_provider="ollama", llm_model="llama3.3")
    assert config.llm_max_concurrent_requests == 8


async def test_stage_routed_local_gets_local_cap(monkeypatch):
    """A stage routed to a local server carries the local cap on its own
    config even when the base model is cloud — stage_config() re-applies
    _apply_local_rate_limit_default, so no dedicated stage fields are needed.
    """
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=None,
        LLM_RATE_LIMIT_ENABLED="false",
        LLM_EXTRACTION_MODEL="llama3.3",
        LLM_EXTRACTION_PROVIDER="ollama",
    )
    config = _build(llm_provider="openai", llm_model="gpt-4o-mini")
    stage = config.stage_config("extraction")
    assert stage.llm_max_concurrent_requests == LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS


async def test_unset_cap_lets_unbounded_requests_run(monkeypatch):
    """Cap disabled: 10 simultaneous requests all proceed (fast path)."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=0,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    counter = _Counter()
    gate = _Gate()
    held = [_hold_open(_seam, gate, counter) for _ in range(10)]
    try:
        gate.open_for(1.0)
        await asyncio.wait_for(asyncio.gather(*(h.entered() for h in held)), timeout=10)
        assert counter.max_seen == 10
    finally:
        for h in held:
            gate.open_for(2.0)
        await asyncio.wait_for(asyncio.gather(*(h.release() for h in held)), timeout=10)


async def test_cap_holds_back_extra_requests(monkeypatch):
    """Cap 2: the 3rd and 4th requests stay outside the seam until slots free."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=2,
        LLM_RATE_LIMIT_ENABLED="false",
    )
    counter = _Counter()
    gate = _Gate()
    held = [_hold_open(_seam, gate, counter) for _ in range(4)]
    try:
        gate.open_for(1.0)
        await asyncio.wait_for(asyncio.gather(*(h.entered() for h in held[:2])), timeout=10)
        # The extras must still be queued: give them many scheduler slices.
        for _ in range(50):
            await asyncio.sleep(0)
        assert counter.max_seen == 2
        # Free one slot; exactly one extra gets in, no more.
        await asyncio.wait_for(held[0].release(), timeout=10)
        await asyncio.wait_for(held[2].entered(), timeout=10)
        for _ in range(50):
            await asyncio.sleep(0)
        assert counter.max_seen == 2
    finally:
        gate.open_for(2.0)
        await asyncio.wait_for(asyncio.gather(*(h.release() for h in held[1:])), timeout=10)


async def test_backends_do_not_share_slots(monkeypatch):
    """Two models hit their own caps; a capped backend never starves another.

    This is the objection that sank the single-global-gateway semaphore in
    PR #4321: one slow backend must not queue requests aimed at others.
    """
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=1,
        LLM_RATE_LIMIT_ENABLED="false",
    )

    slow = _build(llm_provider="ollama", llm_model="slow-llama")
    fast = _build(llm_provider="openai", llm_model="gpt-4o-mini")

    slow_gate = _Gate()
    fast_gate = _Gate()

    slow_counter = _Counter()
    fast_counter = _Counter()

    held_slow_1 = _hold_open(_seam, slow_gate, slow_counter, config=slow)
    held_slow_2 = _hold_open(_seam, slow_gate, slow_counter, config=slow)
    held_fast_1 = _hold_open(_seam, fast_gate, fast_counter, config=fast)
    try:
        fast_gate.open_for(2.0)
        # The fast backend passes immediately despite the slow one being capped.
        await asyncio.wait_for(held_fast_1.entered(), timeout=10)
        assert fast_counter.max_seen == 1

        slow_gate.open_for(2.0)
        await asyncio.wait_for(held_slow_1.entered(), timeout=10)
        for _ in range(50):
            await asyncio.sleep(0)
        assert slow_counter.max_seen == 1
        assert fast_counter.max_seen == 1
    finally:
        await asyncio.wait_for(
            asyncio.gather(*(h.release() for h in (held_slow_1, held_slow_2, held_fast_1))),
            timeout=10,
        )


async def test_slots_released_on_error(monkeypatch):
    """A failing dispatch gives its slot back (retry loops must not leak)."""
    _set_env(
        monkeypatch,
        LLM_MAX_CONCURRENT_REQUESTS=1,
        LLM_RATE_LIMIT_ENABLED="false",
    )

    async def _fail_once():
        try:
            async with _seam():
                raise RuntimeError("429 rate limit")
        except RuntimeError:
            pass

    await asyncio.wait_for(asyncio.gather(*[_fail_once() for _ in range(5)]), timeout=10)

    # A slot is definitely free: a held request still gets in.
    counter = _Counter()
    gate = _Gate()
    held = _hold_open(_seam, gate, counter)
    try:
        gate.open_for(1.0)
        await asyncio.wait_for(held.entered(), timeout=10)
        assert counter.max_seen == 1
    finally:
        gate.open_for(2.0)
        await asyncio.wait_for(held.release(), timeout=10)
