"""Factory for the litellm_native structured-output client.

Called by ``LLMGateway`` when ``STRUCTURED_OUTPUT_FRAMEWORK="litellm_native"``.
Unlike the instructor factory there is no provider dispatch — one universal
adapter serves every provider — and construction is pure attribute assignment
(no client build, no I/O), so no caching layer is needed: each call reads the
active (possibly per-request) config and builds a fresh adapter.
"""

from functools import lru_cache

from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)

# Providers that do not require ``llm_api_key`` (mirrors get_llm_client): Bedrock
# authenticates with AWS credentials and llama.cpp runs locally.
_NO_API_KEY_PROVIDERS = {"bedrock", "llama_cpp"}

# cognee keeps provider and model as separate settings, and the instructor path
# did its own per-provider dispatch. litellm instead routes on a
# provider-qualified model name, so a config that is perfectly valid for
# instructor -- LLM_PROVIDER=ollama with LLM_MODEL=phi4 -- reaches litellm as a
# bare "phi4" and dies with:
#
#     litellm.BadRequestError: LLM Provider NOT provided. You passed model=phi4
#
# Only providers whose litellm prefix is unambiguous are listed. openai and azure
# are deliberately absent: litellm already resolves bare OpenAI model names, and
# azure needs a deployment-specific form we must not guess at.
_LITELLM_PROVIDER_PREFIX = {
    "ollama": "ollama",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "mistral": "mistral",
    "bedrock": "bedrock",
}


@lru_cache(maxsize=256)
def _qualify_model(model: str, provider: str) -> str:
    """Prefix ``model`` with its litellm provider only when litellm needs it.

    Deliberately conservative: an already-qualified name ("ollama/phi4") and any
    bare name litellm can already resolve on its own are returned untouched, so
    this cannot re-route a configuration that works today. It only rescues the
    case that currently raises BadRequestError before a request is even sent.

    A slash does not by itself mean the name carries a provider. Ollama accepts
    namespaced names ("library/phi4") and Hugging Face GGUF paths
    ("hf.co/user/repo"), so litellm decides instead.
    """
    if not model:
        return model

    import litellm

    try:
        litellm.get_llm_provider(model=model)
        return model
    except Exception:
        prefix = _LITELLM_PROVIDER_PREFIX.get((provider or "").lower())
        return f"{prefix}/{model}" if prefix else model


def get_native_client(raise_api_key_error: bool = True) -> NativeLiteLLMAdapter:
    """Build a ``NativeLiteLLMAdapter`` from the active LLM configuration.

    Args:
        raise_api_key_error: When ``True`` (default), raise ``LLMAPIKeyNotSetError``
            if no API key is configured for a provider that needs one. Set
            ``False`` where the key may be legitimately absent (e.g. computing
            chunk sizes).
    """
    llm_config = get_llm_context_config()

    # Provider-aware key requirement (Azure may authenticate via managed identity).
    api_key = llm_config.llm_api_key
    provider = (llm_config.llm_provider or "").lower()
    requires_api_key = provider not in _NO_API_KEY_PROVIDERS and not (
        provider == "azure" and llm_config.llm_azure_use_managed_identity
    )
    if raise_api_key_error and requires_api_key and (api_key is None or api_key.strip() == ""):
        raise LLMAPIKeyNotSetError()

    # Cap generation at the model's ceiling when LiteLLM knows it, else the user's
    # configured limit — same computation as get_llm_client.
    from cognee.infrastructure.llm.utils import get_model_max_completion_tokens

    model_max = get_model_max_completion_tokens(llm_config.llm_model)
    user_max = llm_config.llm_max_completion_tokens
    max_completion_tokens = min(model_max, user_max) if model_max is not None else user_max

    return NativeLiteLLMAdapter(
        api_key=api_key or "",
        model=_qualify_model(llm_config.llm_model, provider),
        max_completion_tokens=max_completion_tokens,
        endpoint=llm_config.llm_endpoint or None,
        api_version=llm_config.llm_api_version,
        fallback_model=llm_config.fallback_model or None,
        fallback_api_key=llm_config.fallback_api_key or None,
        fallback_endpoint=llm_config.fallback_endpoint or None,
        llm_args=llm_config.llm_args or None,
    )
