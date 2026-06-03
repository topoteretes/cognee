"""Regression tests for LLM call hangs and preflight routing.

Covers:
  (b) #2456/#2796/#2902 — a structured-output / completion call against an endpoint
      that accepts the connection but never responds must raise promptly instead of
      hanging forever. ``LLMGateway.acreate_structured_output`` now wraps the call in
      a bounded ``asyncio.wait_for``.
  (a) #2752 — the preflight ``test_llm_connection`` must treat a structured-output /
      parse failure from a *reachable* (non-OpenAI/local) endpoint as healthy, while
      still failing on genuine timeouts / connection errors.
"""

import asyncio
import importlib

import pytest

from cognee.infrastructure.llm import utils as llm_utils
from cognee.infrastructure.llm.LLMGateway import LLMGateway

# The package __init__ re-exports the ``LLMGateway`` *class* under the same dotted
# name as the submodule, so a plain ``import ... as`` yields the class. Load the
# real module object explicitly so we can monkeypatch module-level globals.
gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")


class _StubConfig:
    def __init__(self, timeout):
        self.structured_output_framework = "instructor"
        self.llm_call_timeout_seconds = timeout
        self.llm_model = "openai/gpt-test"


class _HangingClient:
    """LLM client whose structured-output call never returns."""

    async def acreate_structured_output(self, *args, **kwargs):
        # Simulate an endpoint that accepted the connection but never responds.
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_gateway_structured_output_does_not_hang(monkeypatch):
    """A hanging endpoint must raise TimeoutError quickly, not hang."""
    monkeypatch.setattr(gateway_module, "get_llm_config", lambda: _StubConfig(timeout=1))

    # Patch the lazily-imported get_llm_client used inside acreate_structured_output.
    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client as glc

    monkeypatch.setattr(glc, "get_llm_client", lambda *a, **k: _HangingClient())

    coro = LLMGateway.acreate_structured_output(
        text_input="hi",
        system_prompt="sys",
        response_model=str,
    )

    # Outer guard (8s) is far larger than the configured 1s LLM timeout. If the
    # gateway bound is missing the call hangs and the *outer* guard fires near the
    # 8s mark; with the bound in place the gateway raises near the 1s mark. We
    # assert the failure surfaces well before the guard, proving the bound works
    # rather than masking a hang. (asyncio.TimeoutError is a subclass of
    # TimeoutError on 3.11+, so timing is what distinguishes the two paths.)
    loop = asyncio.get_event_loop()
    start = loop.time()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(coro, timeout=8)
    elapsed = loop.time() - start
    assert elapsed < 5, (
        f"Call took {elapsed:.1f}s; the gateway timeout bound did not fire "
        "promptly (likely missing)."
    )


@pytest.mark.asyncio
async def test_gateway_timeout_disabled_with_non_positive(monkeypatch):
    """A non-positive timeout disables the bound (escape hatch)."""
    monkeypatch.setattr(gateway_module, "get_llm_config", lambda: _StubConfig(timeout=0))

    class _FastClient:
        async def acreate_structured_output(self, *args, **kwargs):
            return "ok"

    import cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client as glc

    monkeypatch.setattr(glc, "get_llm_client", lambda *a, **k: _FastClient())

    result = await LLMGateway.acreate_structured_output(
        text_input="hi",
        system_prompt="sys",
        response_model=str,
    )
    assert result == "ok"


@pytest.mark.asyncio
async def test_preflight_tolerates_structured_output_failure(monkeypatch):
    """#2752: a reachable endpoint that fails the structured-output round-trip
    (e.g. cannot coerce a bare ``str`` model) must NOT fail the preflight."""

    async def _raise_parse_error(*args, **kwargs):
        raise ValueError("could not parse response into str")

    monkeypatch.setattr(gateway_module.LLMGateway, "acreate_structured_output", _raise_parse_error)

    # Should return without raising.
    await llm_utils.test_llm_connection()


@pytest.mark.asyncio
async def test_preflight_still_fails_on_timeout(monkeypatch):
    """Genuine unresponsiveness must still surface as a TimeoutError."""

    monkeypatch.setattr(llm_utils, "CONNECTION_TEST_TIMEOUT_SECONDS", 1)

    async def _hang(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(gateway_module.LLMGateway, "acreate_structured_output", _hang)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(llm_utils.test_llm_connection(), timeout=10)
