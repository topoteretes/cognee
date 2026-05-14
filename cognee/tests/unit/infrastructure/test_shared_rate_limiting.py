import importlib

from cognee.infrastructure.llm.config import get_llm_config


def test_default_llm_rate_limit_is_below_gpt_5_5_tier_1():
    get_llm_config.cache_clear()

    config = get_llm_config()

    assert config.llm_rate_limit_enabled is True
    assert config.llm_rate_limit_requests == 450
    assert config.llm_rate_limit_interval == 60


def test_shared_llm_limiter_uses_llm_interval(monkeypatch):
    import cognee.shared.rate_limiting as rate_limiting

    with monkeypatch.context() as patched_env:
        patched_env.setenv("LLM_RATE_LIMIT_ENABLED", "true")
        patched_env.setenv("LLM_RATE_LIMIT_REQUESTS", "7")
        patched_env.setenv("LLM_RATE_LIMIT_INTERVAL", "11")
        patched_env.setenv("EMBEDDING_RATE_LIMIT_INTERVAL", "29")
        get_llm_config.cache_clear()

        rate_limiting = importlib.reload(rate_limiting)

        assert rate_limiting.llm_rate_limiter.max_rate == 7
        assert rate_limiting.llm_rate_limiter.time_period == 11
        assert rate_limiting.embedding_rate_limiter.time_period == 29

    get_llm_config.cache_clear()
    importlib.reload(rate_limiting)
