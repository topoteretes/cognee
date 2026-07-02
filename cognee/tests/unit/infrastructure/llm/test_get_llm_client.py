from dataclasses import asdict

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm import utils as llm_utils
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api import (
    adapter as generic_adapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    _LLM_CLIENT_CACHE_MAXSIZE,
    LLMProvider,
    _build_llm_client_cache_key,
    _freeze_for_cache,
    _get_llm_client_cached,
    _raise_for_missing_api_key,
    _unfreeze_from_cache,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai import (
    adapter as openai_adapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
    OpenAIAdapter,
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


def test_get_max_chunk_tokens_prefers_model_context_window(monkeypatch):
    class FakeEmbeddingEngine:
        max_completion_tokens = 4096

    class FakeVectorEngine:
        embedding_engine = FakeEmbeddingEngine()

    class FakeLLMClient:
        max_completion_tokens = 1024

    class FakeLLMConfig:
        llm_model = "test-model"

    monkeypatch.setattr(
        "cognee.infrastructure.databases.vector.get_vector_engine",
        lambda: FakeVectorEngine(),
    )
    monkeypatch.setattr(llm_utils, "get_llm_client", lambda raise_api_key_error=False: FakeLLMClient())
    monkeypatch.setattr(llm_utils, "get_llm_context_config", lambda: FakeLLMConfig())
    monkeypatch.setattr(llm_utils, "get_model_max_context_tokens", lambda model_name: 32768)

    assert llm_utils.get_max_chunk_tokens() == 4096


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


def test_openai_adapter_preserves_default_instructor_mode_for_non_gpt5_models(monkeypatch):
    calls = []

    def fake_from_litellm(completion, **kwargs):
        calls.append((completion, kwargs))
        return object()

    monkeypatch.setattr(generic_adapter.instructor, "from_litellm", fake_from_litellm)
    monkeypatch.setattr(openai_adapter.instructor, "from_litellm", fake_from_litellm)

    OpenAIAdapter(api_key="test-key", model="openai/gpt-4o-mini", max_completion_tokens=1024)

    assert calls[-2][1] == {}
    assert calls[-1][1] == {}


def test_openai_adapter_honors_explicit_instructor_mode_for_non_gpt5_models(monkeypatch):
    calls = []

    def fake_from_litellm(completion, **kwargs):
        calls.append((completion, kwargs))
        return object()

    monkeypatch.setattr(generic_adapter.instructor, "from_litellm", fake_from_litellm)
    monkeypatch.setattr(openai_adapter.instructor, "from_litellm", fake_from_litellm)

    OpenAIAdapter(
        api_key="test-key",
        model="openai/qwen36-35b-a3b-fp8-gpu0",
        max_completion_tokens=1024,
        instructor_mode="json_mode",
    )

    assert calls[-2][1]["mode"].value == "json_mode"
    assert calls[-1][1]["mode"].value == "json_mode"


@pytest.mark.asyncio
async def test_openai_adapter_passes_fallback_endpoint_on_policy_fallback(monkeypatch):
    class PolicyFallbackTrigger(Exception):
        pass

    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise PolicyFallbackTrigger("blocked")
            return "fallback-response"

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(openai_adapter, "ContentFilterFinishReasonError", PolicyFallbackTrigger)
    monkeypatch.setattr(
        generic_adapter.instructor, "from_litellm", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(openai_adapter.instructor, "from_litellm", lambda *args, **kwargs: object())

    adapter = OpenAIAdapter(
        api_key="primary-key",
        model="openai/gpt-5-mini",
        endpoint="https://primary.example.test",
        max_completion_tokens=1024,
        fallback_api_key="fallback-key",
        fallback_endpoint="https://fallback.example.test",
        fallback_model="openai/gpt-5-mini-fallback",
    )
    adapter.aclient = FakeClient()

    result = await adapter.acreate_structured_output("input", "system", BaseModel)

    assert result == "fallback-response"
    assert calls[0]["api_base"] == "https://primary.example.test"
    assert calls[1]["api_base"] == "https://fallback.example.test"
