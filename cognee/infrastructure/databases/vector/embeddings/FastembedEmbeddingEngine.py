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
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.embeddings.utils import (
    handle_embedding_response,
    sanitize_embedding_text_inputs,
)
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer
from cognee.shared.logging_utils import get_logger
from cognee.shared.model_download_notice import log_model_load
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager

litellm.set_verbose = False
logger = get_logger("FastembedEmbeddingEngine")


def fastembed_model_cached(model: str) -> tuple[bool, str, str | None]:
    """Whether fastembed already holds ``model`` locally, its cache dir, and a size hint.

    fastembed resolves its cache the same way: ``FASTEMBED_CACHE_PATH`` or
    ``fastembed_cache`` under the system temp dir. A model downloaded from the
    hub lives under ``models--<repo>``; one fetched from fastembed's own
    bucket under ``fast-<name>``. Zero-network; unknown models report a
    download with no size.
    """
    cache_dir = Path(
        os.getenv("FASTEMBED_CACHE_PATH", os.path.join(tempfile.gettempdir(), "fastembed_cache"))
    )
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
    - MAX_RETRIES: The maximum number of retries for embedding operations.
    """

    model: str
    dimensions: int
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self,
        model: str | None = "openai/text-embedding-3-large",
        dimensions: int | None = 3072,
        max_completion_tokens: int = 512,
        batch_size: int = 100,
    ):
        self.model = model
        self.dimensions = dimensions
        self.max_completion_tokens = max_completion_tokens
        self.tokenizer = self.get_tokenizer()
        self.batch_size = batch_size
        cached, cache_dir, size_hint = fastembed_model_cached(model)
        log_model_load(
            logger,
            model=model,
            cached=cached,
            cache_dir=cache_dir,
            size_hint=size_hint,
            location_var="FASTEMBED_CACHE_PATH",
        )
        self.embedding_model = TextEmbedding(model_name=model)

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

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
    async def embed_text(self, text: list[str]) -> list[list[float]]:
        """
        Embed the given text into numerical vectors.

        This method generates embeddings for a list of text strings. If mocking is enabled, it
        returns zero vectors instead. It handles exceptions by logging the error and raising an
        `EmbeddingException` on failure.

        Parameters:
        -----------

            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[List[float]]: A list of embeddings, where each embedding is a list of floats
              representing the vector form of the input text.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        try:
            if self.mock:
                embeddings = [[0.0] * self.dimensions for _ in sanitized_text]
            else:
                async with embedding_rate_limiter_context_manager():
                    # fastembed/onnxruntime inference is CPU-bound and synchronous; run it in a
                    # worker thread so it doesn't block the event loop while batches embed.
                    embeddings = await asyncio.to_thread(
                        self.embedding_model.embed,
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
                if len(original_texts) > 1:
                    mid = math.ceil(len(original_texts) / 2)
                    left_vecs, right_vecs = await asyncio.gather(
                        self.embed_text(original_texts[:mid]),
                        self.embed_text(original_texts[mid:]),
                    )
                    embeddings = left_vecs + right_vecs
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

                if len(original_texts) == 1:
                    s = original_texts[0]
                    third = len(s) // 3
                    if third == 0:
                        raise EmbeddingContextWindowTooSmallError from error
                    left_part, right_part = s[: third * 2], s[third:]
                    (left_vec,), (right_vec,) = await asyncio.gather(
                        self.embed_text([left_part]),
                        self.embed_text([right_part]),
                    )
                    pooled = (np.array(left_vec) + np.array(right_vec)) / 2
                    embeddings = [pooled.tolist()]
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

                return handle_embedding_response(original_texts, embeddings, self.dimensions)

            logger.error(f"Embedding error in FastembedEmbeddingEngine: {error!s}")
            raise EmbeddingException(
                f"Failed to index data points using model {self.model}"
            ) from error

        return handle_embedding_response(original_texts, embeddings, self.dimensions)

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
        tokenizer = resolve_embedding_tokenizer(
            provider="fastembed",
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
        )
        logger.debug("Tokenizer loaded for FastembedEmbeddingEngine")
        return tokenizer
