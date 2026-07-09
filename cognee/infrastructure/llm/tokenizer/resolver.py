"""Resolve the tokenizer that best matches a configured embedding model.

Chunk sizing counts tokens with a tokenizer. When that tokenizer does not match
the embedding model, the counts are wrong: a BERT wordpiece model (BGE, MiniLM,
E5, ...) counted with the OpenAI BPE tokenizer, for example, mis-sizes every
chunk and skews the ``--dry-run`` token estimate. See issue #3646.

This module centralizes model to tokenizer resolution so all embedding engines
pick the closest tokenizer the same way, with one safe fallback and one clear,
advisory warning when a mismatch is unavoidable.

Resolution, by embedding provider:

* ``openai``            -> ``TikTokenTokenizer`` for the model (BPE, correct).
* ``gemini``            -> ``TikTokenTokenizer`` default (Gemini has no local
  tokenizer; token counts are approximate).
* ``mistral``           -> ``MistralTokenizer`` for the model.
* ``fastembed``         -> the model's own HuggingFace tokenizer (BGE/MiniLM are
  wordpiece), instead of the old hardcoded ``gpt-4o`` BPE tokenizer.
* ollama / openai-compatible / custom / other -> an explicit
  ``HUGGINGFACE_TOKENIZER`` override if set, otherwise the embedding model's own
  HuggingFace repo.

Any failure to load a matching tokenizer falls back to TikToken and logs a
warning. Resolution is advisory only and never raises: a wrong count is a
degraded estimate, not a fatal error.
"""

from typing import Callable, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.Mistral import MistralTokenizer
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.infrastructure.llm.tokenizer.tokenizer_interface import TokenizerInterface

logger = get_logger("tokenizer_resolver")

# Appended to every mismatch warning so the operator knows the impact and the fix.
_MISMATCH_HINT = (
    "Token counts drive chunk sizing and the --dry-run estimate, so a tokenizer "
    "that does not match the embedding model will mis-size chunks. Set "
    "HUGGINGFACE_TOKENIZER to a tokenizer matching your embedding model to fix this."
)


def _bare_model(model: Optional[str]) -> Optional[str]:
    """Drop a single leading ``provider/`` tag from a model id.

    Splits once, so a multi-segment repo after the tag survives
    (``openai/text-embedding-3-large`` -> ``text-embedding-3-large``;
    ``hosted_vllm/BAAI/bge-m3`` -> ``BAAI/bge-m3``). Only applied to providers
    whose model id carries a provider prefix (openai, mistral); HuggingFace-repo
    providers keep the full id.
    """
    return model.split("/", 1)[-1] if model and "/" in model else model


def _fastembed_hf_repo(model: Optional[str]) -> Optional[str]:
    """Return the HuggingFace repo whose tokenizer matches a fastembed model.

    fastembed model ids are the canonical HF repos (``BAAI/bge-small-en-v1.5``,
    ``sentence-transformers/all-MiniLM-L6-v2``), which carry the right wordpiece
    tokenizer. Returns the matching id, or ``None`` when the model is unknown.
    """
    if not model:
        return None

    try:
        from fastembed import TextEmbedding

        supported = TextEmbedding.list_supported_models()
    except Exception:
        # fastembed not installed here (e.g. CI unit tests): best-effort treat a
        # namespaced id as an HF repo, otherwise give up so the caller warns.
        return model if "/" in model else None

    bare = _bare_model(model)
    for entry in supported:
        name = entry.get("model", "")
        if name == model or _bare_model(name) == bare:
            return name
    return None


def _load_or_tiktoken_fallback(
    build: Callable[[], TokenizerInterface],
    max_completion_tokens: int,
    *,
    context: str,
) -> TokenizerInterface:
    """Build a tokenizer via ``build``; on any failure warn and fall back to the
    default TikToken tokenizer.

    This is where the module's "never raises" guarantee is enforced: every
    tokenizer constructor that can throw is routed through here. That includes
    ``tiktoken.encoding_for_model`` raising ``KeyError`` on a model it does not
    know, a HuggingFace repo failing to load, and ``MistralTokenizer`` raising
    when ``mistral-common`` is not installed. A failure degrades to an
    approximate count rather than aborting chunk sizing.
    """
    try:
        return build()
    except Exception as error:
        logger.warning(
            "Could not load a matching tokenizer for %s (%s). Falling back to "
            "TikToken, so token counts are approximate. %s",
            context,
            error,
            _MISMATCH_HINT,
        )
        return TikTokenTokenizer(model=None, max_completion_tokens=max_completion_tokens)


def _huggingface_or_fallback(
    repo: str,
    max_completion_tokens: int,
    *,
    context: str,
) -> TokenizerInterface:
    """Load a HuggingFace tokenizer for ``repo``; fall back to TikToken with a warning."""
    return _load_or_tiktoken_fallback(
        lambda: HuggingFaceTokenizer(model=repo, max_completion_tokens=max_completion_tokens),
        max_completion_tokens,
        context=context,
    )


def resolve_embedding_tokenizer(
    *,
    provider: Optional[str],
    model: Optional[str],
    max_completion_tokens: int = 512,
    huggingface_tokenizer: Optional[str] = None,
) -> TokenizerInterface:
    """Resolve the tokenizer that best matches an embedding provider and model.

    Parameters:
        provider: Embedding provider (``openai``, ``gemini``, ``mistral``,
            ``fastembed``, ``ollama``, ``openai_compatible``, ``custom``, ...).
        model: Embedding model id, optionally provider-prefixed.
        max_completion_tokens: Passed through to the tokenizer.
        huggingface_tokenizer: Explicit ``HUGGINGFACE_TOKENIZER`` override, used
            for providers whose model id is not itself a HuggingFace repo.

    Returns a ready tokenizer. Never raises: on any failure it warns and returns
    a TikToken fallback.
    """
    provider_lower = (provider or "").lower()
    bare = _bare_model(model)

    if "openai" in provider_lower and "compatible" not in provider_lower:
        # tiktoken.encoding_for_model raises KeyError on a model it does not know
        # (e.g. a newly released embedding model), so guard the "never raises"
        # contract with the shared warn-and-fall-back-to-default-TikToken path.
        return _load_or_tiktoken_fallback(
            lambda: TikTokenTokenizer(model=bare, max_completion_tokens=max_completion_tokens),
            max_completion_tokens,
            context=f"openai embedding model {model!r}",
        )

    if "gemini" in provider_lower:
        # Gemini tokenization needs a network call; approximate locally with TikToken.
        return TikTokenTokenizer(model=None, max_completion_tokens=max_completion_tokens)

    if "mistral" in provider_lower:
        # MistralTokenizer raises if mistral-common is not installed (optional
        # dependency), so route it through the same safe fallback.
        return _load_or_tiktoken_fallback(
            lambda: MistralTokenizer(model=bare, max_completion_tokens=max_completion_tokens),
            max_completion_tokens,
            context=f"mistral embedding model {model!r}",
        )

    if "fastembed" in provider_lower:
        repo = _fastembed_hf_repo(model)
        if repo is None:
            logger.warning(
                "Fastembed model %r is not in the known model map, so tokens are counted "
                "with TikToken (BPE) rather than the model's own tokenizer. %s",
                model,
                _MISMATCH_HINT,
            )
            return TikTokenTokenizer(model=None, max_completion_tokens=max_completion_tokens)
        return _huggingface_or_fallback(
            repo, max_completion_tokens, context=f"fastembed model {model!r}"
        )

    # ollama / openai-compatible / custom / other: an explicit HUGGINGFACE_TOKENIZER
    # override wins, otherwise use the embedding model's own repo.
    target = huggingface_tokenizer or model
    if not target:
        logger.warning(
            "No tokenizer could be resolved for provider=%r model=%r; counting with TikToken. %s",
            provider,
            model,
            _MISMATCH_HINT,
        )
        return TikTokenTokenizer(model=None, max_completion_tokens=max_completion_tokens)

    if huggingface_tokenizer and model and _bare_model(huggingface_tokenizer) != bare:
        # An override that plausibly differs from the model is legitimate (an
        # Ollama model id is not an HF repo), but flag it at warning level so a
        # genuine mismatch is not silent (the fallback paths also warn, and
        # logger.info is invisible under the default log config).
        logger.warning(
            "Counting tokens for embedding model %r with HUGGINGFACE_TOKENIZER=%r. "
            "Make sure they share a tokenizer, otherwise chunk sizing may be off.",
            model,
            huggingface_tokenizer,
        )

    return _huggingface_or_fallback(
        target, max_completion_tokens, context=f"embedding model {target!r}"
    )
