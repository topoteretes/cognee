"""Factory for the litellm_native structured-output client.

Called by ``LLMGateway`` when ``STRUCTURED_OUTPUT_FRAMEWORK="litellm_native"``.
Unlike the instructor factory there is no provider dispatch — one universal
adapter serves every provider — and construction is pure attribute assignment
(no client build, no I/O), so no caching layer is needed: each call reads the
active (possibly per-request) config and builds a fresh adapter.
"""

from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)


def get_native_client(raise_api_key_error: bool = True) -> NativeLiteLLMAdapter:
    """Build a ``NativeLiteLLMAdapter`` from the active LLM configuration.

    Args:
        raise_api_key_error: When ``True`` (default), raise ``LLMAPIKeyNotSetError``
            if no API key is configured. Set ``False`` where the key may be
            legitimately absent (e.g. computing chunk sizes).
    """
    llm_config = get_llm_context_config()

    api_key = llm_config.llm_api_key
    if raise_api_key_error and (api_key is None or api_key.strip() == ""):
        raise LLMAPIKeyNotSetError()

    # Cap generation length at the model's ceiling when LiteLLM knows it,
    # mirroring get_llm_client.
    from cognee.infrastructure.llm.utils import get_model_max_completion_tokens

    model_max = get_model_max_completion_tokens(llm_config.llm_model)
    user_max = llm_config.llm_max_completion_tokens
    max_completion_tokens = min(model_max, user_max) if model_max is not None else user_max

    return NativeLiteLLMAdapter(
        api_key=api_key or "",
        model=llm_config.llm_model,
        max_completion_tokens=max_completion_tokens,
        endpoint=llm_config.llm_endpoint or None,
        api_version=llm_config.llm_api_version,
        fallback_model=llm_config.fallback_model or None,
        fallback_api_key=llm_config.fallback_api_key or None,
        fallback_endpoint=llm_config.fallback_endpoint or None,
        llm_args=llm_config.llm_args or None,
    )
