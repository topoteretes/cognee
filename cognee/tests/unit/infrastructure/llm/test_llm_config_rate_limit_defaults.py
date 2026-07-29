"""Tests for LLMConfig.default_local_rate_limit_budget (provider-aware RPM default)."""

import pytest

from cognee.infrastructure.llm.config import LOCAL_DEFAULT_RATE_LIMIT_REQUESTS, LLMConfig


def _build(**kwargs):
    defaults = dict(llm_api_key="test-key", llm_endpoint="http://localhost:11434/v1")
    defaults.update(kwargs)
    return LLMConfig(**defaults)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("ollama", "phi4:latest"),
        ("llama_cpp", "some-model"),
        ("custom", "lm_studio/qwen2.5-7b"),
        ("custom", "hosted_vllm/meta-llama/Llama-3-70B"),
        ("custom", "vllm/some-model"),
    ],
)
def test_local_providers_default_to_small_budget(provider, model):
    config = _build(llm_provider=provider, llm_model=model)
    assert config.llm_rate_limit_requests == LOCAL_DEFAULT_RATE_LIMIT_REQUESTS


def test_cloud_providers_keep_regular_default():
    config = _build(llm_provider="openai", llm_model="openai/gpt-5-mini")
    assert config.llm_rate_limit_requests == 60


def test_explicit_setting_wins_over_local_default():
    config = _build(llm_provider="ollama", llm_model="phi4:latest", llm_rate_limit_requests=90)
    assert config.llm_rate_limit_requests == 90
