from types import SimpleNamespace

from cognee.infrastructure.llm.utils import resolve_llm_token_limit


def _llm_config(**overrides):
    config = {
        "llm_provider": "openai",
        "llm_max_completion_tokens": 16384,
        "ollama_num_ctx": 4096,
        "llama_cpp_n_ctx": 2048,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_resolve_llm_token_limit_uses_ollama_num_ctx_when_model_unknown():
    config = _llm_config(llm_provider="ollama", ollama_num_ctx=4096)

    assert resolve_llm_token_limit("llama3.1:8b", config) == 4096


def test_resolve_llm_token_limit_uses_llama_cpp_n_ctx_when_model_unknown():
    config = _llm_config(llm_provider="llama_cpp", llama_cpp_n_ctx=2048)

    assert resolve_llm_token_limit("local-model", config) == 2048


def test_resolve_llm_token_limit_respects_user_ceiling_for_ollama():
    config = _llm_config(
        llm_provider="ollama",
        llm_max_completion_tokens=2048,
        ollama_num_ctx=4096,
    )

    assert resolve_llm_token_limit("llama3.1:8b", config) == 2048


def test_resolve_llm_token_limit_falls_back_to_user_max_for_known_providers():
    config = _llm_config(llm_max_completion_tokens=8192)

    assert resolve_llm_token_limit("openai/unknown-model-not-in-litellm", config) == 8192
