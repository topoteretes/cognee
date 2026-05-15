from dataclasses import asdict

import pytest

from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    _LLM_CLIENT_CACHE_MAXSIZE,
    LLMProvider,
    _build_llm_client_cache_key,
    _freeze_for_cache,
    _get_llm_client_cached,
    _raise_for_missing_api_key,
    _unfreeze_from_cache,
)


def _llm_config(**overrides):
    config = {
        "llm_provider": "openai",
        "llm_model": "openai/gpt-5-mini",
        "llm_api_key": "primary-secret",
        "llm_endpoint": "https://llm.example.test",
        "llm_api_version": "2024-01-01",
        "llm_instructor_mode": "TOOLS",
        "llm_streaming": False,
        "llm_max_completion_tokens": 2048,
        "transcription_model": "whisper-1",
        "fallback_api_key": "fallback-secret",
        "fallback_endpoint": "https://fallback.example.test",
        "fallback_model": "fallback-model",
        "llm_azure_use_managed_identity": False,
        "llm_args": {"extra_body": {"strict": True}, "temperature": 0},
        "llama_cpp_model_path": "/models/test.gguf",
        "llama_cpp_n_ctx": 2048,
        "llama_cpp_n_gpu_layers": 1,
        "llama_cpp_chat_format": "chatml",
    }
    config.update(overrides)
    return LLMConfig(**config)


def test_llm_client_cache_is_bounded_lru():
    assert _get_llm_client_cached.cache_parameters()["maxsize"] == _LLM_CLIENT_CACHE_MAXSIZE


def test_llm_client_cache_key_covers_adapter_configuration_fields():
    base = _build_llm_client_cache_key(_llm_config(), max_completion_tokens=1024)

    changed_keys = [
        _build_llm_client_cache_key(_llm_config(llm_api_key="other-secret"), 1024),
        _build_llm_client_cache_key(_llm_config(llm_api_version="2024-02-01"), 1024),
        _build_llm_client_cache_key(_llm_config(llm_streaming=True), 1024),
        _build_llm_client_cache_key(_llm_config(fallback_api_key="other-fallback"), 1024),
        _build_llm_client_cache_key(_llm_config(fallback_endpoint="https://other.test"), 1024),
        _build_llm_client_cache_key(_llm_config(fallback_model="other-model"), 1024),
        _build_llm_client_cache_key(_llm_config(llm_args={"temperature": 1}), 1024),
        _build_llm_client_cache_key(_llm_config(llm_azure_use_managed_identity=True), 1024),
        _build_llm_client_cache_key(_llm_config(), max_completion_tokens=4096),
    ]

    assert all(cache_key != base for cache_key in changed_keys)


def test_llm_client_cache_key_does_not_expose_raw_secrets():
    cache_key = _build_llm_client_cache_key(_llm_config(), max_completion_tokens=1024)

    rendered_key = repr(cache_key)
    rendered_dataclass = str(asdict(cache_key))

    assert "primary-secret" not in rendered_key
    assert "fallback-secret" not in rendered_key
    assert "primary-secret" not in rendered_dataclass
    assert "fallback-secret" not in rendered_dataclass
    assert not hasattr(cache_key.api_key_cache_key, "__dict__")
    assert not hasattr(cache_key.fallback_api_key_cache_key, "__dict__")


def test_llm_args_cache_representation_is_stable_and_rebuildable():
    llm_args = {
        "z": [3, {"nested": True}],
        "a": {"b": (1, 2), "c": {"x", "y"}},
    }
    first = _freeze_for_cache(llm_args)
    second = _freeze_for_cache(
        {
            "a": {"c": {"y", "x"}, "b": (1, 2)},
            "z": [3, {"nested": True}],
        }
    )

    assert first == second
    assert _unfreeze_from_cache(first) == llm_args


def test_api_key_validation_still_happens_before_cache_lookup():
    with pytest.raises(LLMAPIKeyNotSetError):
        _raise_for_missing_api_key(LLMProvider.OPENAI, None, raise_api_key_error=True)

    with pytest.raises(LLMAPIKeyNotSetError):
        _raise_for_missing_api_key(LLMProvider.ANTHROPIC, "", raise_api_key_error=True)

    with pytest.raises(LLMAPIKeyNotSetError):
        _raise_for_missing_api_key(
            LLMProvider.AZURE,
            "   ",
            raise_api_key_error=True,
            use_managed_identity=False,
        )

    _raise_for_missing_api_key(LLMProvider.OPENAI, None, raise_api_key_error=False)
    _raise_for_missing_api_key(LLMProvider.BEDROCK, None, raise_api_key_error=True)
    _raise_for_missing_api_key(
        LLMProvider.AZURE,
        None,
        raise_api_key_error=True,
        use_managed_identity=True,
    )
