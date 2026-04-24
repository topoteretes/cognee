"""
Embedding engine for OpenAI-compatible API servers (llama.cpp, vLLM, TEI, etc.).

Uses the openai SDK directly instead of litellm to avoid known incompatibilities
with local embedding servers:
- litellm sends ``encoding_format: null`` which strict JSON parsers reject
  (llama.cpp: ``[json.exception.type_error.302] type must be string, but is null``)
- litellm's response parsing breaks with ``encoding_format: "float"`` on custom
  ``api_base`` endpoints (``'list' object has no attribute 'model_dump'``)

See: https://docs.litellm.ai/blog/vllm-embeddings-incident
     https://github.com/BerriAI/litellm/issues/19174
"""

import asyncio
import logging
import math
import os
from typing import List, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_delay,
    wait_exponential_jitter,
)

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
)
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.infrastructure.databases.vector.embeddings.utils import (
    handle_embedding_response,
    sanitize_embedding_text_inputs,
)
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager
from cognee.shared.logging_utils import get_logger

logger = get_logger("OpenAICompatibleEmbeddingEngine")


class EmbeddingException(Exception):
    """Raised when an embedding request fails."""

    def __init__(self, message: str, name: str = "EmbeddingException"):
        self.message = message
        self.name = name
        super().__init__(self.message)


class OpenAICompatibleEmbeddingEngine(EmbeddingEngine):
    """
    Embedding engine for any server that exposes the OpenAI ``/v1/embeddings`` API.

    Designed for local/self-hosted servers such as:
    - llama.cpp (``llama-server --embedding``)
    - vLLM
    - Hugging Face TEI
    - LocalAI, Infinity, etc.

    Unlike :class:`LiteLLMEmbeddingEngine`, this engine communicates with the server
    directly via the ``openai`` Python SDK, avoiding litellm's parameter injection
    and response-parsing issues with custom endpoints.

    Public methods:
    - embed_text: Embed a list of strings into vector representations.
    - get_vector_size: Retrieve the size of the embedding vectors.
    - get_batch_size: Return the configured batch size.
    """

    model: str
    dimensions: int
    max_completion_tokens: int
    endpoint: str
    api_key: str
    mock: bool
    batch_size: int
    tokenizer: object

    def __init__(
        self,
        model: Optional[str] = "default",
        dimensions: int = 3072,
        max_completion_tokens: int = 8191,
        endpoint: Optional[str] = "http://localhost:8080",
        api_key: Optional[str] = "no-key-required",
        batch_size: int = 36,
    ):
        self.model = model or "default"
        self.dimensions = dimensions
        self.max_completion_tokens = max_completion_tokens
        self.endpoint = endpoint or "http://localhost:8080"
        self.api_key = api_key or "no-key-required"
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false").lower()
        self.mock = enable_mocking in ("true", "1", "yes")

        # Normalise the base URL: the openai SDK appends /embeddings automatically,
        # so we need the URL to end with /v1 (not /v1/embeddings).
        base = self.endpoint.rstrip("/")
        if base.endswith("/v1/embeddings"):
            base = base[: -len("/embeddings")]
        if not base.endswith("/v1"):
            base = base + "/v1"
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=base)

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type((ValueError, EmbeddingException)),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vector representations.

        If the input exceeds the model's context window, the method will recursively
        split the input and combine the results.

        Parameters:
        -----------

            - text (List[str]): A list of strings to be embedded.

        Returns:
        --------

            - List[List[float]]: A list of vectors representing the embedded texts.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        if self.mock:
            embeddings = [[0.0] * (self.dimensions or 1) for _ in sanitized_text]
            return handle_embedding_response(original_texts, embeddings, self.dimensions)

        try:
            async with embedding_rate_limiter_context_manager():
                response = await asyncio.wait_for(
                    self._client.embeddings.create(
                        model=self.model,
                        input=sanitized_text,
                        encoding_format="float",
                    ),
                    timeout=30.0,
                )
            embeddings = [item.embedding for item in response.data]

        except Exception as error:
            error_str = str(error).lower()

            # Handle context window exceeded by splitting input
            context_error_patterns = (
                "context length",
                "context window",
                "too long",
                "maximum context",
                "maximum tokens",
                "max tokens",
            )
            if any(pattern in error_str for pattern in context_error_patterns):
                if isinstance(original_texts, list) and len(original_texts) > 1:
                    mid = math.ceil(len(original_texts) / 2)
                    left_vecs, right_vecs = await asyncio.gather(
                        self.embed_text(original_texts[:mid]),
                        self.embed_text(original_texts[mid:]),
                    )
                    embeddings = left_vecs + right_vecs
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

                if isinstance(original_texts, list) and len(original_texts) == 1:
                    s = original_texts[0]
                    third = len(s) // 3
                    if third == 0:
                        raise EmbeddingException(
                            "Text is too short to split further but exceeds context window."
                        ) from error
                    left_part, right_part = s[: third * 2], s[third:]
                    (left_vec,), (right_vec,) = await asyncio.gather(
                        self.embed_text([left_part]),
                        self.embed_text([right_part]),
                    )
                    pooled = (np.array(left_vec) + np.array(right_vec)) / 2
                    embeddings = [pooled.tolist()]
                    return handle_embedding_response(original_texts, embeddings, self.dimensions)

            if isinstance(error, asyncio.TimeoutError):
                logger.error(
                    "Embedding endpoint timed out. EMBEDDING_ENDPOINT='%s'.",
                    self.endpoint,
                )
                raise EmbeddingException(
                    "Embedding request timed out. Check EMBEDDING_ENDPOINT connectivity."
                ) from error

            if isinstance(error, (httpx.ConnectError, httpx.ReadTimeout)):
                logger.error(
                    "Failed to connect to embedding endpoint. EMBEDDING_ENDPOINT='%s'.",
                    self.endpoint,
                )
                raise EmbeddingException(
                    "Cannot connect to embedding endpoint. Check EMBEDDING_ENDPOINT."
                ) from error

            logger.error(
                "Error embedding text: %s. EMBEDDING_ENDPOINT='%s'.",
                str(error),
                self.endpoint,
            )
            raise EmbeddingException(
                "Embedding failed. Verify EMBEDDING_ENDPOINT and server status."
            ) from error

        return handle_embedding_response(original_texts, embeddings, self.dimensions)

    def get_vector_size(self) -> int:
        """
        Retrieve the size of the embedding vectors.

        Returns:
        --------

            - int: The dimension of the embedding vectors.
        """
        return self.dimensions

    def get_batch_size(self) -> int:
        """
        Return the desired batch size for embedding calls.

        Returns:
        --------

            - int: The batch size.
        """
        return self.batch_size

    def get_tokenizer(self):
        """Load a tokenizer for chunk sizing against OpenAI-compatible embedding servers."""
        logger.debug("Loading HuggingfaceTokenizer for OpenAICompatibleEmbeddingEngine...")
        try:
            tokenizer = HuggingFaceTokenizer(
                model=self.model,
                max_completion_tokens=self.max_completion_tokens,
            )
        except Exception as error:
            logger.warning("Could not get tokenizer from HuggingFace due to: %s", error)
            logger.info("Switching to TikToken default tokenizer.")
            tokenizer = TikTokenTokenizer(
                model=None, max_completion_tokens=self.max_completion_tokens
            )
        return tokenizer
