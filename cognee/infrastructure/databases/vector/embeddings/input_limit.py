"""The number of tokens an embedding model accepts, and the limit chunks are sized by.

Every embedding engine sizes chunks (and everything else it embeds) by
``max_completion_tokens``. That value used to be whatever the config said,
regardless of the model, and the config default was OpenAI's 8191. Models that
accept less truncate the input without an error, so on fastembed's default model
(512 tokens) every chunk was embedded from its first 512 tokens only.

The effective limit is now ``min(cap, model limit)``: the cap is
``EMBEDDING_MAX_COMPLETION_TOKENS`` (``DEFAULT_EMBEDDING_INPUT_CAP`` when unset),
the model limit comes from the best source each provider has:

* fastembed: the loaded model's own tokenizer truncation;
* litellm providers: litellm's model table (``max_input_tokens``);
* Ollama: the model's ``context_length`` from ``/api/show``;
* any HuggingFace-repo model: the tokenizer's ``model_max_length``, as a fallback.

When no source knows the model, the cap stands and an INFO line says the limit
could not be verified.
"""

import litellm

from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# The chunk-token cap when EMBEDDING_MAX_COMPLETION_TOKENS is not set. Below every
# hosted embedding model's limit (OpenAI 8191, Mistral 8192, Voyage 32000) and
# lowered further to the model's own limit when that is known and smaller.
DEFAULT_EMBEDDING_INPUT_CAP = 4096

# Tokenizers with no real limit report a sentinel (transformers uses 2**31 or
# 1e30). Anything this large is not a model limit.
_MAX_PLAUSIBLE_LIMIT = 1_000_000

_OLLAMA_SHOW_TIMEOUT_SECONDS = 3.0


def sane_limit(value) -> int | None:
    """``value`` as a token limit, or None when it is not a plausible one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0 or value > _MAX_PLAUSIBLE_LIMIT:
        return None
    return value


def litellm_input_limit(model: str | None, provider: str | None = None) -> int | None:
    """The model's ``max_input_tokens`` from litellm's model table, or None.

    Tried as given, then as ``provider/model`` and as the bare name after the
    provider prefix, because the table keys some models one way and some the other
    (``azure/text-embedding-3-large`` but plain ``text-embedding-3-large``).
    """
    if not model:
        return None
    candidates = [model]
    if provider:
        candidates.append(f"{provider}/{model}")
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
    for candidate in candidates:
        try:
            info = litellm.get_model_info(candidate)
        except Exception:
            # litellm raises for an unknown model. That is the normal case for
            # a self-hosted one, so it stays at debug.
            logger.debug("litellm has no model info for %r", candidate, exc_info=True)
            continue
        limit = sane_limit(info.get("max_input_tokens")) or sane_limit(info.get("max_tokens"))
        if limit is not None:
            return limit
    return None


def huggingface_tokenizer_limit(tokenizer) -> int | None:
    """``model_max_length`` of a resolved HuggingFace tokenizer, or None.

    Only a HuggingFaceTokenizer carries the model's own limit; TikToken and
    Mistral fallbacks say nothing about the embedding model.
    """
    if not isinstance(tokenizer, HuggingFaceTokenizer):
        return None
    return sane_limit(getattr(tokenizer.tokenizer, "model_max_length", None))


def fastembed_input_limit(embedding_model) -> int | None:
    """The limit a loaded fastembed model truncates at, from its own tokenizer, or None."""
    tokenizer = getattr(getattr(embedding_model, "model", None), "tokenizer", None)
    truncation = getattr(tokenizer, "truncation", None)
    if not isinstance(truncation, dict):
        return None
    return sane_limit(truncation.get("max_length"))


def ollama_input_limit(endpoint: str | None, model: str | None, api_key: str | None) -> int | None:
    """The model's context length from Ollama's ``/api/show``, or None.

    Ollama embeds up to the model's ``context_length`` (its ``num_ctx`` default for
    embeddings). Anything that stops the lookup, from a server that is down to an
    endpoint that is not Ollama-shaped, gives None: the cap then stands, and the
    ``truncate: false`` on every embed request still turns an over-length input
    into an error instead of a silent cut.
    """
    if not endpoint or not model or "/api/" not in endpoint:
        return None
    import httpx

    from cognee.shared.utils import create_secure_ssl_context

    show_url = f"{endpoint.split('/api/', 1)[0]}/api/show"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            show_url,
            json={"model": model},
            headers=headers,
            timeout=_OLLAMA_SHOW_TIMEOUT_SECONDS,
            verify=create_secure_ssl_context(),
        )
        response.raise_for_status()
        model_info = response.json().get("model_info") or {}
    except Exception:
        logger.debug(
            "Could not read the context length of %r from %s", model, show_url, exc_info=True
        )
        return None
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return sane_limit(value)
    return None


def effective_input_limit(
    *,
    configured: int | None,
    model_limit: int | None,
    model: str | None,
    source: str,
) -> int:
    """The token limit an engine embeds by: the cap, lowered to the model's limit.

    ``configured`` is ``EMBEDDING_MAX_COMPLETION_TOKENS`` (None when unset, which
    means ``DEFAULT_EMBEDDING_INPUT_CAP``). A configured value above what the
    model accepts is a misconfiguration and is reported as a WARNING; the default
    being lowered is expected and logged at INFO, as is a model whose limit no
    source knows.
    """
    cap = configured if configured else DEFAULT_EMBEDDING_INPUT_CAP

    if model_limit is None:
        logger.info(
            "Could not determine how many tokens embedding model %r accepts; chunks are "
            "limited to %s tokens (EMBEDDING_MAX_COMPLETION_TOKENS). Set it to the "
            "model's input limit if that is lower.",
            model,
            cap,
        )
        return cap

    if cap <= model_limit:
        logger.debug(
            "Embedding model %r accepts %s tokens (%s); chunks are limited to %s tokens.",
            model,
            model_limit,
            source,
            cap,
        )
        return cap

    if configured:
        logger.warning(
            "EMBEDDING_MAX_COMPLETION_TOKENS=%s exceeds what embedding model %r accepts "
            "(%s tokens, %s). Chunks are limited to %s tokens; text beyond that would be "
            "dropped from the embedding.",
            configured,
            model,
            model_limit,
            source,
            model_limit,
        )
    else:
        logger.info(
            "Embedding model %r accepts %s tokens (%s); chunks are limited to that instead "
            "of the default %s.",
            model,
            model_limit,
            source,
            cap,
        )
    return model_limit
