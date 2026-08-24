"""Zero-network provider-configuration preflight.

Catches the two misconfigurations that otherwise surface minutes into a
first ingestion as an opaque downstream failure:

1. **Only the LLM is configured.** ``get_embedding_engine()`` reuses
   ``LLM_API_KEY`` for embeddings when ``EMBEDDING_API_KEY`` is unset — a
   convenience documented for OpenAI keys. With a non-OpenAI
   ``LLM_PROVIDER`` the wrong key (or, for provider ``custom``, no key at
   all) is handed to the default OpenAI embedding model and the first
   embedding call fails mid-cognify with a provider authentication error.
2. **Only embeddings are configured.** The LLM key requirement (mirroring
   ``get_native_client``) is unmet, so every completion call would raise
   ``LLMAPIKeyNotSetError`` after ingestion work has already started.

``check_provider_config`` is a pure function over the two config objects so
it can be unit-tested and reused by ``cognee-cli doctor``;
``validate_provider_config`` is the cheap, once-per-process hot-path wrapper
called at the top of ``add()`` / ``remember()``.
"""

from __future__ import annotations

import os
import threading
from typing import List

from cognee.exceptions import CogneeConfigurationError

# Mirrors get_native_client._NO_API_KEY_PROVIDERS: Bedrock authenticates with
# AWS credentials and llama.cpp runs locally.
_NO_API_KEY_LLM_PROVIDERS = {"bedrock", "llama_cpp"}
_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Env vars that disable the preflight. COGNEE_SKIP_CONNECTION_TEST and
# MOCK_EMBEDDING are honored because environments that set them (CI, offline
# runs) deliberately operate with partial or fake provider config.
_SKIP_ENV_VARS = ("COGNEE_SKIP_PREFLIGHT", "COGNEE_SKIP_CONNECTION_TEST", "MOCK_EMBEDDING")

_checked = False
_check_lock = threading.Lock()


class ProviderConfigMismatchError(CogneeConfigurationError):
    """LLM and embedding provider settings are mutually inconsistent."""

    def __init__(self, message: str):
        super().__init__(message, "ProviderConfigMismatchError")


def check_provider_config(llm_config=None, embedding_config=None) -> List[str]:
    """Return actionable problem descriptions for inconsistent provider config.

    Pure and network-free: only reads attributes off the two config objects.
    An empty list means the LLM/embedding pairing is consistent (it does not
    guarantee the credentials work — that is ``cognee-cli doctor --probe``).
    """
    if llm_config is None:
        from cognee.infrastructure.llm.config import get_llm_context_config

        llm_config = get_llm_context_config()
    if embedding_config is None:
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_context_config,
        )

        embedding_config = get_embedding_context_config()

    problems: List[str] = []

    llm_provider = (llm_config.llm_provider or "").lower()
    llm_key = (llm_config.llm_api_key or "").strip()
    embedding_provider = (embedding_config.embedding_provider or "").lower()
    embedding_key = (embedding_config.embedding_api_key or "").strip()
    embedding_model = embedding_config.embedding_model or ""
    embedding_endpoint = embedding_config.embedding_endpoint

    embeddings_untouched = (
        embedding_provider == "openai"
        and embedding_model == _DEFAULT_EMBEDDING_MODEL
        and not embedding_key
        and not embedding_endpoint
    )

    if embeddings_untouched and llm_provider not in ("", "openai"):
        if llm_provider == "custom":
            key_consequence = (
                "with LLM_PROVIDER='custom' no API key at all would be sent to the "
                "OpenAI embeddings endpoint"
            )
        else:
            key_consequence = (
                f"your LLM_API_KEY (a '{llm_provider}' key) would be sent to the "
                "OpenAI embeddings endpoint"
            )
        problems.append(
            "Embeddings are not configured and would silently default to OpenAI "
            f"('{_DEFAULT_EMBEDDING_MODEL}') while LLM_PROVIDER='{llm_provider}': "
            f"{key_consequence}, so the first ingestion would fail with an "
            "authentication error minutes later. Fix: set EMBEDDING_PROVIDER, "
            "EMBEDDING_MODEL and EMBEDDING_API_KEY for your embedding service "
            "(or EMBEDDING_ENDPOINT for a local one, e.g. Ollama), or provide an "
            "OpenAI EMBEDDING_API_KEY to keep the default embedder."
        )

    llm_requires_key = llm_provider not in _NO_API_KEY_LLM_PROVIDERS and not (
        llm_provider == "azure" and getattr(llm_config, "llm_azure_use_managed_identity", False)
    )

    if not embeddings_untouched and llm_requires_key and not llm_key:
        problems.append(
            "Embedding settings are configured but LLM_API_KEY is not set "
            f"(LLM_PROVIDER='{llm_provider or 'openai'}' requires one). Entity "
            "extraction and search completions all need the LLM, so ingestion "
            "would start and then fail at the first LLM call. Fix: set "
            "LLM_API_KEY (and LLM_PROVIDER/LLM_MODEL if you are not using "
            "OpenAI)."
        )

    return problems


def _skip_preflight() -> bool:
    return any(os.getenv(var, "").lower() in ("true", "1", "yes") for var in _SKIP_ENV_VARS)


def validate_provider_config(force: bool = False) -> None:
    """Raise ``ProviderConfigMismatchError`` when provider config is inconsistent.

    Runs the pure check once per process (subsequent calls are a no-op unless
    ``force=True``); success is cached, failure is not, so a caller that fixes
    the config in-process gets re-validated. No network I/O.
    """
    global _checked
    if _checked and not force:
        return
    with _check_lock:
        if _checked and not force:
            return
        if _skip_preflight():
            _checked = True
            return
        problems = check_provider_config()
        if problems:
            raise ProviderConfigMismatchError("\n".join(problems))
        _checked = True


def reset_preflight_state() -> None:
    """Testing hook: forget that the preflight already ran in this process."""
    global _checked
    _checked = False
