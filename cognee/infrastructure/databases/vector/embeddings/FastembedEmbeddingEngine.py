import asyncio
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    from fastembed import TextEmbedding
except ImportError:
    raise ImportError(
        "fastembed is required for FastembedEmbeddingEngine but is not importable; it is a "
        "core cognee dependency. Reinstall it with: pip install fastembed"
    )

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.databases.exceptions import (
    EmbeddingContextWindowTooSmallError,
    EmbeddingException,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
    EmbeddingResult,
)
from cognee.infrastructure.databases.vector.embeddings.input_limit import (
    init_input_limit,
    sane_limit,
)
from cognee.infrastructure.databases.vector.embeddings.utils import (
    count_input_tokens,
    handle_embedding_response,
    sanitize_embedding_text_inputs,
)
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.model_download_notice import log_model_load
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager

litellm.set_verbose = False
logger = get_logger("FastembedEmbeddingEngine")

observe = get_observe()


def resolve_fastembed_cache_dir() -> str:
    """The directory fastembed keeps its models in, created if it can be.

    Precedence: ``FASTEMBED_CACHE_PATH`` (fastembed's own variable, taken as
    given) over the Hugging Face hub cache (``HF_HUB_CACHE``, else
    ``$HF_HOME/hub``, default ``~/.cache/huggingface/hub``), where the GLiNER
    model is kept too. fastembed downloads through the hub in the same layout,
    so both local models share one cache and one variable moves both. Left to
    itself fastembed would use ``fastembed_cache`` under the system temp
    directory, which is emptied on many hosts at every restart, so the model was
    downloaded again each time. When the hub cache cannot be created or written
    to (a read-only home, a sandbox) the temp location is used after all, with a
    warning, so an embed never fails over where its model is kept.

    Sharing the hub cache has one side effect: when a hub download fails,
    fastembed falls back to its own bucket and unpacks into ``tmp/`` and
    ``fast-<name>/`` at the cache root, which ``hf cache scan`` lists as invalid
    repos. Both belong to fastembed and are safe to leave or delete.
    """
    explicit = os.getenv("FASTEMBED_CACHE_PATH")
    if explicit:
        return explicit

    from huggingface_hub import constants

    cache_dir = Path(constants.HF_HUB_CACHE)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if os.access(cache_dir, os.W_OK):
            return str(cache_dir)
        reason = "not writable"
    except OSError as error:
        reason = str(error)

    fallback = os.path.join(tempfile.gettempdir(), "fastembed_cache")
    logger.warning(
        "fastembed model cache %s is unusable (%s); using %s instead. The model will be "
        "downloaded again after a restart that clears the temp directory. Set HF_HOME "
        "or FASTEMBED_CACHE_PATH to a writable location.",
        cache_dir,
        reason,
        fallback,
    )
    return fallback


def fastembed_model_cached(
    model: str, cache_dir: str | None = None
) -> tuple[bool, str, str | None]:
    """Whether fastembed already holds ``model`` locally, its cache dir, and a size hint.

    ``cache_dir`` defaults to ``resolve_fastembed_cache_dir()``, the directory
    the engine hands fastembed. A model downloaded from the hub lives under
    ``models--<repo>``; one fetched from fastembed's own bucket under
    ``fast-<name>``. Zero-network; unknown models report a download with no size.
    """
    cache_dir = Path(cache_dir or resolve_fastembed_cache_dir())
    description = next(
        (entry for entry in TextEmbedding.list_supported_models() if entry.get("model") == model),
        None,
    )
    size_gb = (description or {}).get("size_in_GB")
    size_hint = f"about {round(size_gb * 1000)} MB" if size_gb else None
    hub_repo = ((description or {}).get("sources") or {}).get("hf")
    candidates = [cache_dir / f"fast-{model.split('/')[-1]}"]
    if hub_repo:
        candidates.append(cache_dir / f"models--{hub_repo.replace('/', '--')}")
    return any(path.exists() for path in candidates), str(cache_dir), size_hint


def pad_to_batch_longest(embedding_model) -> None:
    """Make the loaded model pad every batch to its longest text, as fastembed does
    for models whose tokenizer.json stores no padding.

    Some repos store a fixed-length padding in tokenizer.json (all-MiniLM-L6-v2 and
    gte-base: 128), and fastembed keeps it: texts below the length are padded to it,
    longer ones (up to the model's window) are left as they are, so a batch mixing
    the two is a ragged array and fastembed fails on it. Padding is masked by the
    model, so the embeddings do not change. Truncation is left as fastembed set it.
    """
    tokenizer = getattr(getattr(embedding_model, "model", None), "tokenizer", None)
    padding = getattr(tokenizer, "padding", None)
    if not isinstance(padding, dict) or padding.get("length") is None:
        return
    tokenizer.enable_padding(
        direction=padding["direction"],
        pad_id=padding["pad_id"],
        pad_type_id=padding["pad_type_id"],
        pad_token=padding["pad_token"],
    )


class FastembedEmbeddingEngine(EmbeddingEngine):
    """
    Manages the embedding process using a specified model to generate text embeddings.

    Public methods:

    - embed_text
    - get_vector_size
    - get_tokenizer

    Instance variables:

    - model: The name of the embedding model.
    - dimensions: The dimensionality of the embeddings.
    - mock: A flag indicating whether to use mocking instead of the actual embedding model.
    """

    model: str
    dimensions: int
    mock: bool

    def __init__(
        self,
        model: str | None = "openai/text-embedding-3-large",
        dimensions: int | None = 3072,
        max_completion_tokens: int | None = None,
        batch_size: int = 100,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        cache_dir = resolve_fastembed_cache_dir()
        cached, cache_dir, size_hint = fastembed_model_cached(model, cache_dir)
        log_model_load(
            logger,
            model=model,
            cached=cached,
            cache_dir=cache_dir,
            size_hint=size_hint,
            # Name the variable that decided the location: HF_HOME has no effect
            # while FASTEMBED_CACHE_PATH is set.
            location_var=(
                "FASTEMBED_CACHE_PATH" if os.getenv("FASTEMBED_CACHE_PATH") else "HF_HOME"
            ),
        )
        # Passed explicitly: fastembed's own default is the system temp directory.
        self.embedding_model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        pad_to_batch_longest(self.embedding_model)
        # fastembed truncates input at the model's own limit without an error, so
        # chunks must never be sized past it: see input_limit().
        init_input_limit(self, max_completion_tokens)
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    @observe(as_type="embeddings")
    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (
                EmbeddingContextWindowTooSmallError,
                litellm.exceptions.NotFoundError,
                asyncio.CancelledError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def embed_text(self, text: list[str]) -> EmbeddingResult:
        """
        Embed the given text into numerical vectors; returns them and the input-token count.

        This method generates embeddings for a list of text strings. If mocking is enabled, it
        returns zero vectors instead. It handles exceptions by logging the error and raising an
        `EmbeddingException` on failure.

        Parameters:
        -----------

            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - EmbeddingResult: A vector per input text and the input tokens, counted with
              the model's own tokenizer.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        # fastembed cuts input past the model's limit without an error, so an
        # over-length text is caught here and embedded in overlapping parts, the
        # same recovery the other engines run when their provider rejects it.
        if not self.mock and self._exceeds_model_limit(sanitized_text):
            return await self._embed_in_parts(original_texts)

        try:
            if self.mock:
                embeddings = [[0.0] * self.dimensions for _ in sanitized_text]
            else:
                async with embedding_rate_limiter_context_manager():
                    embeddings = self.embedding_model.embed(
                        sanitized_text,
                        batch_size=len(sanitized_text),
                        parallel=None,
                    )

                embeddings = [e.tolist() for e in embeddings]

        except Exception as error:
            error_str = str(error).lower()
            context_error_patterns = (
                "context length",
                "context window",
                "input length",
                "too long",
                "maximum context",
                "maximum tokens",
                "max tokens",
            )
            if any(pattern in error_str for pattern in context_error_patterns):
                return await self._embed_in_parts(original_texts)

            logger.error(f"Embedding error in FastembedEmbeddingEngine: {error!s}")
            raise EmbeddingException(
                f"Failed to index data points using model {self.model}"
            ) from error

        return EmbeddingResult(
            vectors=handle_embedding_response(original_texts, embeddings, self.dimensions),
            # A mock embeds nothing; fastembed itself reports no usage.
            tokens=0 if self.mock else count_input_tokens(None, sanitized_text, self.tokenizer),
        )

    input_limit_source = "fastembed model tokenizer"

    async def input_limit(self) -> int | None:
        """How many tokens of text the loaded model embeds, from its own tokenizer:
        the length it truncates at, less the special tokens it adds itself, since
        text is counted without them. None if fastembed does not expose it."""
        tokenizer = getattr(getattr(self.embedding_model, "model", None), "tokenizer", None)
        truncation = getattr(tokenizer, "truncation", None)
        if not isinstance(truncation, dict):
            return None
        limit = sane_limit(truncation.get("max_length"))
        if limit is None:
            return None
        return sane_limit(limit - tokenizer.num_special_tokens_to_add(False))

    def _exceeds_model_limit(self, texts: list[str]) -> bool:
        """Whether the model's tokenizer would truncate any of ``texts``.

        The tokenizer keeps the cut-off part as ``overflowing`` pieces, so this
        is the model's own verdict, not an estimate. A tokenizer that cannot
        answer (an unexpected fastembed internals layout) means "no", which keeps
        the pre-fix behaviour rather than raising.
        """
        tokenizer = getattr(getattr(self.embedding_model, "model", None), "tokenizer", None)
        if tokenizer is None:
            return False
        try:
            return any(encoding.overflowing for encoding in tokenizer.encode_batch(texts))
        except Exception:
            logger.debug("Could not check input length with the fastembed tokenizer", exc_info=True)
            return False

    async def _embed_in_parts(self, original_texts: list[str]) -> EmbeddingResult:
        """Embed over-length input in parts: a batch is split in two, a single text
        into two overlapping halves whose embeddings are pooled. Each part goes back
        through :meth:`embed_text`, which splits further while it is still too long.
        """
        if len(original_texts) > 1:
            mid = math.ceil(len(original_texts) / 2)
            left_result, right_result = await asyncio.gather(
                self.embed_text(original_texts[:mid]),
                self.embed_text(original_texts[mid:]),
            )
            return EmbeddingResult(
                vectors=handle_embedding_response(
                    original_texts, left_result + right_result, self.dimensions
                ),
                tokens=left_result.tokens + right_result.tokens,
            )

        s = original_texts[0]
        third = len(s) // 3
        if third == 0:
            raise EmbeddingContextWindowTooSmallError
        logger.debug(
            "Text of %s characters exceeds what %s embeds at once (%s tokens); embedding "
            "it as two overlapping parts.",
            len(s),
            self.model,
            self.max_completion_tokens,
        )
        left_part, right_part = s[: third * 2], s[third:]
        left_result, right_result = await asyncio.gather(
            self.embed_text([left_part]),
            self.embed_text([right_part]),
        )
        (left_vec,), (right_vec,) = left_result, right_result
        pooled = (np.array(left_vec) + np.array(right_vec)) / 2
        # Tokens are what the model processed, overlaps included - as the hosted
        # engines that count what they were billed for; with recursive splitting
        # this can be several times the text's own count.
        return EmbeddingResult(
            vectors=handle_embedding_response(original_texts, [pooled.tolist()], self.dimensions),
            tokens=left_result.tokens + right_result.tokens,
        )

    def get_vector_size(self) -> int:
        """
        Return the size of the embedding vector produced by this engine.

        Returns:
        --------

            - int: The dimensionality of the embedding vectors.
        """
        return self.dimensions

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls

        Returns:

        """
        return self.batch_size

    def get_tokenizer(self):
        """
        Instantiate and return the tokenizer used for preparing text for embedding.

        Resolves the fastembed model's own tokenizer (BGE/MiniLM are wordpiece)
        instead of the OpenAI BPE tokenizer, which mis-counted them (issue #3646).

        Returns:
        --------

            A tokenizer object configured for the specified model and maximum token size.
        """
        logger.debug("Loading tokenizer for FastembedEmbeddingEngine...")
        tokenizer = resolve_embedding_tokenizer(provider="fastembed", model=self.model)
        logger.debug("Tokenizer loaded for FastembedEmbeddingEngine")
        return tokenizer
