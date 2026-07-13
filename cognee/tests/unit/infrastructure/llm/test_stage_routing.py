"""Tests for per-stage LLM model routing.

`LLMConfig.stage_config(stage)` merges `llm_<stage>_*` overrides onto the base
`llm_*` fields, and `pipeline_stage(stage)` applies that merged config onto the
`llm_config` ContextVar for the duration of a pipeline stage. Since
`get_llm_client` reads that same ContextVar (via `get_llm_context_config`) to
build its adapter cache key, a stage with overrides configured gets its own
cached client with no change to the gateway or any task call site.

No network calls: these tests only inspect config merging, ContextVar state,
and cache-key construction.
"""

from unittest.mock import AsyncMock

import pytest

from cognee.context_global_variables import (
    llm_config as llm_config_ctx,
    current_pipeline_stage,
)
from cognee.infrastructure.llm.config import LLMConfig, get_llm_context_config
from cognee.infrastructure.llm.LLMGateway import _record_session_usage_after
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    _build_llm_client_cache_key,
    _get_llm_client_cached,
)


def _base_config(**overrides) -> LLMConfig:
    config = {
        "llm_provider": "openai",
        "llm_model": "base-model",
        "llm_endpoint": "",
        "llm_api_key": "base-key",
    }
    config.update(overrides)
    return LLMConfig(**config)


@pytest.fixture(autouse=True)
def _isolate_stage_routing():
    """Keep each test hermetic: clear the LRU cache and reset the ContextVars."""
    _get_llm_client_cached.cache_clear()
    llm_token = llm_config_ctx.set(None)
    stage_token = current_pipeline_stage.set(None)
    try:
        yield
    finally:
        current_pipeline_stage.reset(stage_token)
        llm_config_ctx.reset(llm_token)
        _get_llm_client_cached.cache_clear()


def test_pipeline_stage_routes_to_the_stage_model():
    """Inside the block the extraction override is active; outside it is not."""
    llm_config_ctx.set(_base_config(llm_extraction_model="extract-model"))

    assert get_llm_context_config().llm_model == "base-model"
    with pipeline_stage("extraction"):
        assert get_llm_context_config().llm_model == "extract-model"
    assert get_llm_context_config().llm_model == "base-model"


def test_pipeline_stage_produces_a_stage_specific_cache_key():
    """Proves get_llm_client would build a distinct client per stage, without
    constructing a real client.
    """
    llm_config_ctx.set(_base_config(llm_extraction_model="extract-model"))

    outside_key = _build_llm_client_cache_key(get_llm_context_config(), max_completion_tokens=1024)
    with pipeline_stage("extraction"):
        inside_key = _build_llm_client_cache_key(
            get_llm_context_config(), max_completion_tokens=1024
        )

    assert outside_key.model == "base-model"
    assert inside_key.model == "extract-model"
    assert inside_key != outside_key


def test_pipeline_stage_is_a_no_op_without_stage_overrides():
    """Locks in single-model behavior: with no stage overrides, the effective
    config and cache key are identical inside and outside the block.
    """
    llm_config_ctx.set(_base_config())

    outside_config = get_llm_context_config()
    outside_key = _build_llm_client_cache_key(outside_config, max_completion_tokens=1024)
    with pipeline_stage("extraction"):
        inside_config = get_llm_context_config()
        inside_key = _build_llm_client_cache_key(inside_config, max_completion_tokens=1024)

    assert inside_config.llm_model == outside_config.llm_model
    assert inside_key == outside_key


@pytest.mark.asyncio
async def test_record_session_usage_after_reads_the_context_config(monkeypatch):
    """Bug fix: usage recording must reflect the stage model, not the global
    default.
    """
    llm_config_ctx.set(_base_config(llm_model="ctx-model"))

    record_llm_call = AsyncMock()
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.usage_tracking.record_llm_call", record_llm_call
    )

    async def _trivial_coro():
        return "result"

    await _record_session_usage_after(_trivial_coro(), text_input="x")

    record_llm_call.assert_called_once()
    assert record_llm_call.call_args.kwargs["model"] == "ctx-model"


def test_pipeline_stage_labels_the_current_stage():
    assert current_pipeline_stage.get() is None
    with pipeline_stage("extraction"):
        assert current_pipeline_stage.get() == "extraction"
    assert current_pipeline_stage.get() is None
