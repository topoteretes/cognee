"""Derive embedding settings from the configured LLM provider.

cognee historically defaulted embeddings to OpenAI regardless of the LLM
provider, silently reusing LLM_API_KEY against api.openai.com. For any
non-OpenAI setup without explicit EMBEDDING_* configuration that meant a
guaranteed mid-cognify failure with a generic connection error. When the
user has not configured embeddings explicitly, we now derive a matching
embedding setup from the LLM provider — or fail fast with one clear fix
when the provider has no embedding API to derive from.
"""

from typing import Optional
from urllib.parse import urlparse

from cognee.infrastructure.databases.exceptions import EmbeddingProviderMismatchError

_NO_EMBEDDING_API_FIX = (
    "Set EMBEDDING_PROVIDER=openai (plus EMBEDDING_API_KEY), or "
    "EMBEDDING_PROVIDER=fastembed for a local, zero-key option."
)

_OLLAMA_DEFAULT_EMBED_ENDPOINT = "http://localhost:11434/api/embed"


def _ollama_embedding_endpoint(llm_endpoint: Optional[str]) -> str:
    """Point embeddings at the same Ollama host the LLM uses.

    LLM_ENDPOINT for Ollama looks like ``http://host:11434/v1`` while the
    embedding API lives at ``http://host:11434/api/embed`` — keep the host,
    swap the path.
    """
    if llm_endpoint:
        parsed = urlparse(llm_endpoint)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/api/embed"
    return _OLLAMA_DEFAULT_EMBED_ENDPOINT


def derive_embedding_settings(
    llm_provider: str,
    llm_endpoint: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> Optional[dict]:
    """Derive embedding settings for an LLM provider with no explicit EMBEDDING_* config.

    Returns a dict with keys ``provider``, ``model``, ``endpoint``, ``api_key``,
    ``dimensions`` and ``huggingface_tokenizer`` (values may be None, meaning
    "use the existing config/default"), or None when the provider is unknown —
    in which case the caller keeps today's behavior.

    Raises:
    -------

        - EmbeddingProviderMismatchError: when the provider has no embedding
          API (anthropic, llama_cpp) or needs deployment-specific settings that
          cannot be guessed (azure, bedrock).
    """
    provider = (llm_provider or "").strip().lower()

    if provider == "openai":
        return {
            "provider": "openai",
            "model": "openai/text-embedding-3-large",
            "endpoint": None,
            "api_key": llm_api_key,
            "dimensions": None,
            "huggingface_tokenizer": None,
        }

    if provider == "gemini":
        return {
            "provider": "gemini",
            "model": "gemini/gemini-embedding-001",
            "endpoint": None,
            "api_key": llm_api_key,
            "dimensions": None,
            "huggingface_tokenizer": None,
        }

    if provider == "mistral":
        return {
            "provider": "mistral",
            "model": "mistral/mistral-embed",
            "endpoint": None,
            "api_key": llm_api_key,
            # mistral-embed is not in litellm's dimension registry.
            "dimensions": 1024,
            "huggingface_tokenizer": None,
        }

    if provider == "ollama":
        # OllamaEmbeddingEngine defaults, pointed at the LLM's host when set.
        return {
            "provider": "ollama",
            "model": "avr/sfr-embedding-mistral:latest",
            "endpoint": _ollama_embedding_endpoint(llm_endpoint),
            "api_key": None,
            "dimensions": 1024,
            "huggingface_tokenizer": "Salesforce/SFR-Embedding-Mistral",
        }

    if provider == "custom":
        # OpenAI-compatible endpoint: assume it also serves /v1/embeddings.
        return {
            "provider": "openai_compatible",
            "model": None,
            "endpoint": llm_endpoint or None,
            "api_key": llm_api_key,
            "dimensions": None,
            "huggingface_tokenizer": None,
        }

    if provider in ("anthropic", "llama_cpp"):
        raise EmbeddingProviderMismatchError(
            f"{provider} has no embedding API. {_NO_EMBEDDING_API_FIX}"
        )

    if provider in ("azure", "bedrock"):
        raise EmbeddingProviderMismatchError(
            f"{provider} embeddings can't be derived from the LLM settings — they "
            "need their own deployment endpoint/region. Set EMBEDDING_PROVIDER, "
            "EMBEDDING_MODEL, EMBEDDING_ENDPOINT and EMBEDDING_API_KEY explicitly."
        )

    return None
