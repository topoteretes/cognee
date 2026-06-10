"""Tests for the LLM "global cache" — the per-context ``llm_config`` ContextVar
override that feeds the LRU-cached LLM client.

The feature (see ``feat: add global context for LLM and embedding settings``)
lets a caller pin an ``LLMConfig`` onto the current async context via
``set_database_global_context_variables(..., llm_config=...)``. ``get_llm_client``
reads that config through ``get_llm_context_config`` and the resulting adapter is
cached by config in ``_get_llm_client_cached``.
"""

import pytest

from cognee.context_global_variables import (
    llm_config as llm_config_ctx,
    set_database_global_context_variables,
)
from cognee.infrastructure.llm.config import LLMConfig, get_llm_context_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    _get_llm_client_cached,
    get_llm_client,
)


# Two plain OpenAI configs that differ only by model, so the test can assert the
# active model changes when a context config is applied.
#
# Dummy API keys are used on purpose: the cache/context behaviour under test does
# not make any network calls, and we never want real secrets in a test fixture.
OPENAI_MODEL = "openai/gpt-5.5"


def _openai_config() -> LLMConfig:
    return LLMConfig(
        llm_provider="openai",
        llm_model=OPENAI_MODEL,
        llm_endpoint="",
        llm_api_key="openai-test-key",
    )


@pytest.fixture(autouse=True)
def _isolate_llm_global_cache():
    """Keep each test hermetic: clear the LRU cache and reset the ContextVar."""
    _get_llm_client_cached.cache_clear()
    token = llm_config_ctx.set(None)
    try:
        yield
    finally:
        llm_config_ctx.reset(token)
        _get_llm_client_cached.cache_clear()


@pytest.mark.asyncio
async def test_set_database_global_context_variables_applies_llm_config(monkeypatch):
    """End-to-end: the public entry point pins the config onto the async context."""
    # Disable backend access control so applying the context only sets the
    # ContextVars (no dataset/database machinery is touched).
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    from uuid import uuid4

    # Before the context is applied the active model is the global one, which is
    # different from the override we are about to set. Pass raise_api_key_error=False
    # so this precondition stays hermetic: it only inspects the model and must not
    # depend on the ambient .env providing a global LLM_API_KEY.
    assert get_llm_client(raise_api_key_error=False).model != OPENAI_MODEL

    async with set_database_global_context_variables(
        "test_dataset", uuid4(), llm_config=_openai_config()
    ):
        client = get_llm_client()
        assert client.model == OPENAI_MODEL
        assert client.endpoint == ""
        assert client.api_key == "openai-test-key"
        assert get_llm_context_config().llm_model == OPENAI_MODEL
