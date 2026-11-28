"""Seam tests: embedding dispatch wires config, pacing, the request cap and its policy."""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

import cognee.infrastructure.databases.vector.embeddings.config as embedding_config_module
import cognee.infrastructure.llm.config as llm_config_module
from cognee.shared import rate_limiting
from cognee.shared.overload_policy import OverloadPolicy
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager


class PacerSpy:
    """Stands in for the AsyncLimiter; records pacing and the free slots at that moment."""

    def __init__(self):
        self.entered = 0
        self.free_slots_on_entry = []

    async def __aenter__(self):
        self.entered += 1
        key = ("embedding_requests", id(asyncio.get_running_loop()))
        entry = rate_limiting._concurrency_state.get(key)
        self.free_slots_on_entry.append(None if entry is None else entry[1]._value)
        return self

    async def __aexit__(self, *args):
        return False


def _embedding_config(**overrides):
    defaults = {
        "embedding_rate_limit_enabled": False,
        "embedding_rate_limit_requests": 1000,
        "embedding_rate_limit_interval": 60,
        "embedding_max_concurrent_requests": 128,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def seam(monkeypatch):
    """Fresh policy, pacer spy, slot state and config stubs for every test."""
    policy = OverloadPolicy(embedding_config_module.embedding_pacing_budget)
    monkeypatch.setattr(embedding_config_module, "embedding_overload_policy", policy)
    spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "_embedding_rate_limiter", spy)
    monkeypatch.setattr(rate_limiting, "_concurrency_state", {})
    state = SimpleNamespace(policy=policy, spy=spy, config=_embedding_config(), auto=True)
    monkeypatch.setattr(embedding_config_module, "get_embedding_config", lambda: state.config)
    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: SimpleNamespace(auto_rate_limit=state.auto)
    )
    return state


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "http://api.test")
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


_SUCCESS = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 4)])


def _compatible_engine(create):
    """A real OpenAI-compatible engine whose client answers through ``create``."""
    from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
        OpenAICompatibleEmbeddingEngine,
    )

    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=4,
        max_completion_tokens=8191,
        endpoint="http://localhost:8099",
        api_key="test-key",
        batch_size=8,
    )
    engine._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return engine


@pytest.mark.asyncio
async def test_unpaced_by_default(seam):
    async with embedding_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 0
    assert seam.policy.is_paced() is False


async def _fail_once(error: Exception) -> None:
    with pytest.raises(type(error)):
        async with embedding_rate_limiter_context_manager():
            raise error


@pytest.mark.asyncio
async def test_rate_limit_starts_pacing_for_the_next_attempt(seam):
    await _fail_once(_rate_limit_error())
    assert seam.policy.is_paced() is True

    async with embedding_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 1


@pytest.mark.asyncio
async def test_auto_rate_limit_off_never_starts_throttling(seam):
    seam.auto = False
    await _fail_once(_rate_limit_error())
    assert seam.policy.is_paced() is False

    async with embedding_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 0


@pytest.mark.asyncio
async def test_cap_lets_two_of_three_requests_in(seam):
    seam.config = _embedding_config(embedding_max_concurrent_requests=2)
    inside = 0
    release = asyncio.Event()

    async def request():
        nonlocal inside
        async with embedding_rate_limiter_context_manager():
            inside += 1
            await release.wait()

    tasks = [asyncio.create_task(request()) for _ in range(3)]
    await asyncio.sleep(0.05)
    assert inside == 2

    release.set()
    await asyncio.gather(*tasks)
    assert inside == 3


@pytest.mark.asyncio
async def test_pacing_happens_before_a_slot_is_taken(seam):
    seam.config = _embedding_config(
        embedding_rate_limit_enabled=True, embedding_max_concurrent_requests=1
    )
    for _ in range(2):
        async with embedding_rate_limiter_context_manager():
            pass
    # Each request finds its one slot still free while it is being paced.
    assert seam.spy.free_slots_on_entry == [1, 1]


@pytest.mark.asyncio
async def test_a_request_queued_when_throttling_starts_is_paced(seam):
    seam.config = _embedding_config(embedding_max_concurrent_requests=1)
    release = asyncio.Event()
    reached = []

    async def request(name):
        async with embedding_rate_limiter_context_manager():
            reached.append(name)
            if name == "first":
                await release.wait()

    first = asyncio.create_task(request("first"))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(request("second"))  # queues for the one slot
    await asyncio.sleep(0.01)

    seam.policy.on_error(_rate_limit_error())  # throttling starts while it waits
    release.set()
    await asyncio.gather(first, second)

    assert reached == ["first", "second"]
    assert seam.spy.entered == 1  # the queued request was paced
    assert seam.spy.free_slots_on_entry == [1]  # and held no slot while paced


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_side", ["embedding", "llm"])
async def test_overload_on_one_side_never_paces_the_other(seam, monkeypatch, failing_side):
    from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

    monkeypatch.setattr(
        llm_config_module,
        "llm_overload_policy",
        OverloadPolicy(llm_config_module.llm_pacing_budget),
    )
    llm_spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "_llm_rate_limiter", llm_spy)
    llm_config = SimpleNamespace(
        auto_rate_limit=True,
        llm_rate_limit_enabled=False,
        llm_rate_limit_requests=60,
        llm_rate_limit_interval=60,
    )
    monkeypatch.setattr(llm_config_module, "get_llm_config", lambda: llm_config)

    seams = {
        "embedding": embedding_rate_limiter_context_manager,
        "llm": llm_rate_limiter_context_manager,
    }
    with pytest.raises(openai.RateLimitError):
        async with seams[failing_side]():
            raise _rate_limit_error()
    for dispatch in seams.values():
        async with dispatch():
            pass

    spies = {"embedding": seam.spy, "llm": llm_spy}
    other_side = "llm" if failing_side == "embedding" else "embedding"
    assert spies[failing_side].entered == 1
    assert spies[other_side].entered == 0


@pytest.mark.asyncio
async def test_simulated_overload_through_an_engine_warns_once_paces_and_lapses(seam, monkeypatch):
    import cognee.shared.overload_policy as overload_policy_module

    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    policy = OverloadPolicy(embedding_config_module.embedding_pacing_budget, cooldown_seconds=0.3)
    monkeypatch.setattr(embedding_config_module, "embedding_overload_policy", policy)
    warnings = []
    monkeypatch.setattr(
        overload_policy_module,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    # Retry-After: 0 lets the engine's real retry run without sleeping.
    replies = [_rate_limit_error(retry_after="0"), _rate_limit_error(retry_after="0"), _SUCCESS]

    async def create(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    engine = _compatible_engine(create)

    await engine.embed_text(["hello"])
    assert len(warnings) == 1  # one throttling period, one warning
    assert seam.spy.entered == 2  # both retried attempts were paced
    assert policy.is_paced() is True

    await asyncio.sleep(0.35)
    assert policy.is_paced() is False  # lapsed after the quiet cooldown

    replies.append(_SUCCESS)
    await engine.embed_text(["hello"])
    assert seam.spy.entered == 2  # back to unpaced


@pytest.mark.asyncio
async def test_engine_retries_after_the_providers_retry_after(seam, monkeypatch):
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    replies = [_rate_limit_error(retry_after="5"), _SUCCESS]

    async def create(**kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    engine = _compatible_engine(create)
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    embed_text = type(engine).embed_text.retry_with(sleep=record_sleep)
    await embed_text(engine, ["hello"])
    assert sleeps == [5.0]


@pytest.mark.asyncio
async def test_engine_requests_beyond_the_cap_wait(seam, monkeypatch):
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    seam.config = _embedding_config(embedding_max_concurrent_requests=1)
    reached = 0
    release = asyncio.Event()

    async def create(**kwargs):
        nonlocal reached
        reached += 1
        await release.wait()
        return _SUCCESS

    engine = _compatible_engine(create)
    calls = [asyncio.create_task(engine.embed_text(["hello"])) for _ in range(2)]
    await asyncio.sleep(0.05)
    assert reached == 1  # the second request waits for the one slot

    release.set()
    await asyncio.gather(*calls)
    assert reached == 2
