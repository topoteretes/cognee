"""The number of tokens an embedding model accepts, and the limit chunks are sized by.

Every embedding engine sizes chunks (and everything else it embeds) by
``max_completion_tokens``. That value used to be whatever the config said,
regardless of the model, and the config default was OpenAI's 8191. Models that
accept less truncate the input without an error, so on fastembed's default model
(512 tokens) every chunk was embedded from its first 512 tokens only.

The effective limit is now ``min(cap, model limit)``: the cap is
``EMBEDDING_MAX_COMPLETION_TOKENS`` (``DEFAULT_EMBEDDING_INPUT_CAP`` when unset),
and each engine reads the model limit from the best source its provider has
(fastembed's loaded tokenizer, Ollama's ``/api/show``, litellm's model table, or
the model's HuggingFace tokenizer). The lookups live with their engines; this
module holds the cap, the litellm table lookup two engines share, and the rule
that combines cap and model limit.

When no source knows the model, the cap stands and a WARNING says the limit
could not be verified.
"""

import litellm

from cognee.shared.logging_utils import get_logger

logger = get_logger()

# The chunk-token cap when EMBEDDING_MAX_COMPLETION_TOKENS is not set. Below every
# hosted embedding model's limit (OpenAI 8191, Mistral 8192, Voyage 32000) and
# lowered further to the model's own limit when that is known and smaller.
DEFAULT_EMBEDDING_INPUT_CAP = 4096


def sane_limit(value) -> int | None:
    """``value`` as a token limit: a positive int, else None."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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
        # A bare name can also match a chat model, whose max_tokens is an
        # output limit; only an embedding entry's input limit counts.
        if info.get("mode") != "embedding":
            continue
        limit = sane_limit(info.get("max_input_tokens"))
        if limit is not None:
            return limit
    return None


def init_input_limit(engine, configured: int | None) -> None:
    """Record the configured cap on a new engine; the model's limit comes later.

    The lookup is async (``EmbeddingEngine.input_limit``), so the constructor only
    keeps the cap: ``max_completion_tokens`` is the cap until ``resolve_input_limit``
    has run, which every chunk-size resolution and the connection preflight do.
    """
    engine.input_cap = configured
    engine.model_input_limit = None
    engine.input_limit_resolved = False
    engine.max_completion_tokens = configured if configured else DEFAULT_EMBEDDING_INPUT_CAP


async def resolve_input_limit(engine) -> int:
    """Ask the engine's provider for the model's limit, once, and apply it.

    Returns the limit the engine embeds by: min(cap, model limit). Idempotent after
    the first call. A provider that cannot tell (None) leaves the cap in place, with
    a WARNING from ``effective_input_limit``.
    """
    if not hasattr(engine, "input_limit_resolved"):
        # An engine that predates input_limit() (a third-party adapter): its
        # max_completion_tokens is whatever it set, and there is nothing to ask.
        return getattr(engine, "max_completion_tokens", DEFAULT_EMBEDDING_INPUT_CAP)
    if not engine.input_limit_resolved:
        engine.model_input_limit = await engine.input_limit()
        engine.max_completion_tokens = effective_input_limit(
            configured=engine.input_cap,
            model_limit=engine.model_input_limit,
            model=engine.model,
            source=engine.input_limit_source,
        )
        engine.input_limit_resolved = True
    return engine.max_completion_tokens


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
    being lowered is expected and logged at INFO; a model whose limit no source
    knows is a WARNING, since it may truncate silently.
    """
    cap = configured if configured else DEFAULT_EMBEDDING_INPUT_CAP

    if model_limit is None:
        # A WARNING, not INFO: a model that accepts less than the cap may cut
        # the input without an error, and only the operator can rule that out.
        logger.warning(
            "Could not determine how many tokens embedding model %r accepts; chunks are "
            "limited to %s tokens (EMBEDDING_MAX_COMPLETION_TOKENS). If the model accepts "
            "fewer, set it to the model's limit, or text beyond it may be dropped.",
            model,
            cap,
        )
        return cap

    if cap <= model_limit:
        logger.debug(
            "Embedding model %r accepts %s tokens of text (%s); chunks are limited to %s tokens.",
            model,
            model_limit,
            source,
            cap,
        )
        return cap

    if configured:
        logger.warning(
            "EMBEDDING_MAX_COMPLETION_TOKENS=%s exceeds what embedding model %r accepts "
            "(%s tokens of text, %s). Chunks are limited to %s tokens; text beyond that would be "
            "dropped from the embedding.",
            configured,
            model,
            model_limit,
            source,
            model_limit,
        )
    else:
        logger.info(
            "Embedding model %r accepts %s tokens of text (%s); chunks are limited to that "
            "instead of the default %s.",
            model,
            model_limit,
            source,
            cap,
        )
    return model_limit
