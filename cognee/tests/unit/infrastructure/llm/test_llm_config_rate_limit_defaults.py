"""Tests for LLMConfig.default_local_rate_limit_budget (provider-aware RPM default)."""

import pytest

from cognee.infrastructure.llm.config import LOCAL_DEFAULT_RATE_LIMIT_REQUESTS, LLMConfig


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    """Build every config in this file from kwargs alone, never the ambient env.

    Same trap as test_llm_config.py: cognee loads .env into os.environ at
    import, and pydantic-settings fills any field not passed as a kwarg from
    the environment, so ensure_env_vars_for_ollama sees an ambient env var the
    same as a kwarg. A developer whose .env sets only LLM_API_KEY therefore
    gets "some but not all of the required environment variables" on every
    ollama case here, while CI (which sets all three) passes. Clearing all
    three makes the file hermetic either way.
    """
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_ENDPOINT", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _build(**kwargs):
    defaults = dict(
        llm_api_key="test-key",
        llm_endpoint="http://localhost:11434/v1",
        _env_file=None,
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("ollama", "phi4:latest"),
        ("llama_cpp", "some-model"),
        ("custom", "lm_studio/qwen2.5-7b"),
    ],
)
def test_local_providers_default_to_small_budget(provider, model):
    config = _build(llm_provider=provider, llm_model=model)
    assert config.llm_rate_limit_requests == LOCAL_DEFAULT_RATE_LIMIT_REQUESTS


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "openai/gpt-5-mini"),
        ("custom", "hosted_vllm/meta-llama/Llama-3-70B"),  # vLLM batches like a cloud endpoint
        ("custom", "vllm/some-model"),
    ],
)
def test_regular_providers_keep_regular_default(provider, model):
    config = _build(llm_provider=provider, llm_model=model)
    assert config.llm_rate_limit_requests == 60


def test_explicit_setting_wins_over_local_default():
    config = _build(llm_provider="ollama", llm_model="phi4:latest", llm_rate_limit_requests=90)
    assert config.llm_rate_limit_requests == 90


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("ollama", "phi4:latest", True),
        ("llama_cpp", "some-model", True),
        ("custom", "lm_studio/qwen2.5-7b", True),
        ("custom", "hosted_vllm/meta-llama/Llama-3-70B", False),
        ("openai", "openai/gpt-5-mini", False),
        (None, None, False),
    ],
)
def test_is_local_llm(provider, model, expected):
    from cognee.infrastructure.llm.config import is_local_llm

    assert is_local_llm(provider, model) is expected
