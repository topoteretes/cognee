"""Tests for the LLM in-flight concurrency cap (LLM_MAX_CONCURRENT_REQUESTS).

The cap lives on the same pacing seam as the RPM limiter
(``cognee.shared.rate_limiting``): every dispatch attempt holds a semaphore
slot while the call is in flight. Keying follows the pipeline-stage machinery:
stage-scoped caps bind to the model configured for that stage, calls outside
any stage bind to the base model, and the global cap always applies on top.
"""

import asyncio

import pytest

from cognee.context_global_variables import llm_config as llm_config_ctx
from cognee.infrastructure.llm.config import (
    LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS,
    LLMConfig,
)
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.shared import rate_limiting

# Each mock dispatch stays "in flight" this long, so overlapping attempts are
# unambiguous at millisecond scheduling granularity while the whole suite
# still runs in well under a second.
HOLD_S = 0.05


def _build(**kwargs):
    defaults = {
        "llm_api_key": "test-key",
        "llm_endpoint": "http://localhost:11434/v1",
        "_env_file": None,
    }
    defaults.update(kwargs)
    return LLMConfig(**defaults)


@pytest.fixture(autouse=True)
def _clean_state():
    """Concurrency state and cached configs never leak between tests."""
    rate_limiting._concurrency_state.clear()
    yield
    rate_limiting._concurrency_state.clear()


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class _Gate:
    """Tracks in-flight dispatches made through the pacing seam."""

    def __init__(self):
        self.in_flight = 0
        self.peak = 0

    async def hold(self):
        async with rate_limiting.llm_rate_limiter_context_manager():
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(HOLD_S)
            self.in_flight -= 1

    async def fan_out(self, n):
        await asyncio.gather(*(self.hold() for _ in range(n)))
        return self.peak


async def _run_with_config(config, n_tasks):
    token = llm_config_ctx.set(config)
    try:
        return await _Gate().fan_out(n_tasks)
    finally:
        llm_config_ctx.reset(token)


async def _run_stage(config, stage, n_tasks, extra_tasks=()):
    """n_tasks dispatches inside ``pipeline_stage(stage)`` plus any others."""
    token = llm_config_ctx.set(config)
    try:
        gate = _Gate()

        async def inside():
            with pipeline_stage(stage):
                await gate.hold()

        tasks = [asyncio.create_task(inside()) for _ in range(n_tasks)]
        tasks += [asyncio.create_task(fn(gate)) for fn in extra_tasks]
        await asyncio.gather(*tasks)
        return gate.peak
    finally:
        llm_config_ctx.reset(token)


@pytest.mark.asyncio
async def test_default_is_unbounded():
    config = _build(llm_provider="openai", llm_model="openai/gpt-5.6-luna")
    assert config.llm_max_concurrent_requests == 0
    peak = await _run_with_config(config, 5)
    assert peak == 5  # no cap -> everything overlaps


@pytest.mark.asyncio
async def test_local_provider_gets_small_default_cap():
    config = _build(llm_provider="ollama", llm_model="phi4:latest")
    assert config.llm_max_concurrent_requests == LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS


@pytest.mark.asyncio
async def test_local_default_cap_enforced_but_overridable():
    ollama = _build(llm_provider="ollama", llm_model="phi4:latest")
    peak = await _run_with_config(ollama, 5)
    assert peak == LOCAL_DEFAULT_MAX_CONCURRENT_REQUESTS
    explicit = _build(
        llm_provider="ollama",
        llm_model="phi4:latest",
        llm_max_concurrent_requests=0,  # operator opts back out
    )
    assert await _run_with_config(explicit, 5) == 5


@pytest.mark.asyncio
async def test_global_cap_limits_in_flight():
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_max_concurrent_requests=2,
    )
    peak = await _run_with_config(config, 6)
    assert peak == 2


@pytest.mark.asyncio
async def test_stage_cap_applies_within_stage_only():
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_extraction_model="openai/gpt-mini",
        llm_extraction_max_concurrent_requests=1,
    )
    # Two stage calls share ONE slot: never overlap.
    stage_peak = await _run_stage(config, "extraction", 2)
    assert stage_peak == 1
    # A stage call plus an outside call (different identity, no global cap):
    # the outside call is unconstrained and may overlap the stage call.
    mixed = await _run_stage(config, "extraction", 1, extra_tasks=[lambda gate: gate.hold()])
    assert mixed == 2


@pytest.mark.asyncio
async def test_stage_cap_follows_the_stage_model():
    # Extraction and summarization routed to the SAME model share that
    # model's slots; the cap protects the serving backend, not the stage name.
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_extraction_model="openai/gpt-mini",
        llm_summarization_model="openai/gpt-mini",
        llm_extraction_max_concurrent_requests=1,
        llm_summarization_max_concurrent_requests=1,
    )
    token = llm_config_ctx.set(config)
    try:
        gate = _Gate()

        async def inside(stage):
            with pipeline_stage(stage):
                await gate.hold()

        await asyncio.gather(*(inside(s) for s in ("extraction", "summarization")))
        assert gate.peak == 1
    finally:
        llm_config_ctx.reset(token)


@pytest.mark.asyncio
async def test_global_cap_binds_across_stages_and_identities():
    # No stage cap set: stage calls fold into the global semaphore (no
    # redundant second lock), and the global cap still spans identities.
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_extraction_model="openai/gpt-mini",
        llm_max_concurrent_requests=1,
    )
    peak = await _run_stage(config, "extraction", 2, extra_tasks=[lambda gate: gate.hold()])
    assert peak == 1


@pytest.mark.asyncio
async def test_stage_cap_tighter_than_global_both_held():
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_extraction_model="openai/gpt-mini",
        llm_max_concurrent_requests=5,
        llm_extraction_max_concurrent_requests=1,
    )
    peak = await _run_stage(config, "extraction", 3)
    assert peak == 1  # tighter stage cap wins inside the stage


@pytest.mark.asyncio
async def test_slots_release_even_when_dispatch_raises():
    config = _build(
        llm_provider="openai",
        llm_model="openai/gpt-5.6-luna",
        llm_max_concurrent_requests=1,
    )
    token = llm_config_ctx.set(config)
    try:
        for _ in range(3):  # serial: any leaked slot would deadlock on #2
            with pytest.raises(RuntimeError):
                async with rate_limiting.llm_rate_limiter_context_manager():
                    raise RuntimeError("dispatch failed")
        # A subsequent success proves no slot leaked.
        async with rate_limiting.llm_rate_limiter_context_manager():
            pass
    finally:
        llm_config_ctx.reset(token)
