import asyncio
import math
import os
import logging
from typing import List, Optional

import aiohttp
import litellm
import numpy as np
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager
from cognee.shared.utils import create_secure_ssl_context
from cognee.infrastructure.databases.vector.embeddings.utils import (
    sanitize_embedding_text_inputs,
    handle_embedding_response,
)

logger = get_logger("TwelveLabsEmbeddingEngine")

# Marengo is a multimodal model: the same vector space is shared by text,
# image, audio and video, so text embedded here is directly comparable to
# embeddings of ingested video. marengo3.0 produces 512-dim vectors.
DEFAULT_MODEL = "marengo3.0"
DEFAULT_DIMENSIONS = 512
DEFAULT_ENDPOINT = "https://api.twelvelabs.io/v1.3/embed"


class TwelveLabsEmbeddingEngine(EmbeddingEngine):
    """
    Implements an embedding engine backed by TwelveLabs' Marengo multimodal model.

    Marengo embeds text into the same 512-dimensional space as video, image and
    audio, which lets cognee store text queries and ingested video in one vector
    store and search across them. This engine handles the text side; video is
    embedded through TwelveLabs' video tasks at ingestion time.

    Public methods:
    - embed_text
    - get_vector_size
    - get_batch_size
    - get_tokenizer

    Instance variables:
    - model
    - dimensions
    - max_completion_tokens
    - endpoint
    - api_key
    - mock
    """

    model: str
    dimensions: int
    max_completion_tokens: int
    endpoint: str
    api_key: Optional[str]
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = DEFAULT_MODEL,
        dimensions: Optional[int] = DEFAULT_DIMENSIONS,
        max_completion_tokens: int = 512,
        endpoint: Optional[str] = DEFAULT_ENDPOINT,
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        self.model = model or DEFAULT_MODEL
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.max_completion_tokens = max_completion_tokens
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        # Falls back to the standalone TwelveLabs env var so the key does not
        # have to be threaded through EMBEDDING_API_KEY when only video is used.
        self.api_key = api_key or os.getenv("TWELVELABS_API_KEY")
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Generate Marengo embedding vectors for a list of text prompts.

        If mocking is enabled, returns a list of zero vectors instead of actual
        embeddings.

        Parameters:
        -----------

            - text (List[str]): A list of text prompts for which to generate embeddings.

        Returns:
        --------

            - List[List[float]]: A list of embedding vectors corresponding to the text
              prompts.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        if self.mock:
            embeddings = [[0.0] * self.dimensions for _ in sanitized_text]
            return handle_embedding_response(original_texts, embeddings, self.dimensions)

        if not self.api_key:
            raise EmbeddingException(
                "TwelveLabs API key is missing. Set EMBEDDING_API_KEY or TWELVELABS_API_KEY. "
                "Get a free key at https://twelvelabs.io"
            )

        try:
            embeddings = await asyncio.gather(
                *[self._get_embedding(prompt) for prompt in sanitized_text]
            )
            return handle_embedding_response(original_texts, embeddings, self.dimensions)
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

                return handle_embedding_response(original_texts, embeddings, self.dimensions)

            logger.error(f"Embedding error in TwelveLabsEmbeddingEngine: {str(error)}")
            raise EmbeddingException(
                f"Failed to index data points using model {self.model}"
            ) from error

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type(
            (litellm.exceptions.NotFoundError, ValueError, asyncio.CancelledError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_embedding(self, prompt: str) -> List[float]:
        """
        Call the TwelveLabs /embed endpoint for a single text prompt.

        The endpoint accepts multipart/form-data (``model_name`` + ``text``) and
        returns ``text_embedding.segments[0].float``.
        """
        # The /embed endpoint only accepts multipart/form-data. aiohttp encodes a
        # FormData of plain strings as application/x-www-form-urlencoded unless at
        # least one field carries an explicit content_type, so set one to force
        # the multipart boundary encoding.
        form = aiohttp.FormData()
        form.add_field("model_name", self.model, content_type="text/plain")
        form.add_field("text", prompt, content_type="text/plain")

        headers = {"x-api-key": self.api_key}

        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with embedding_rate_limiter_context_manager():
                async with session.post(
                    self.endpoint, data=form, headers=headers, timeout=60.0
                ) as response:
                    data = await response.json()

                    if "text_embedding" not in data:
                        message = data.get("message", data)
                        msg_str = str(message).lower()
                        if "too long" in msg_str or ("max" in msg_str and "token" in msg_str):
                            raise ValueError(f"Text too long for embedding model: {message}")
                        raise RuntimeError(f"TwelveLabs embedding API error: {message}")

                    segments = data["text_embedding"].get("segments", [])
                    if not segments or "float" not in segments[0]:
                        raise ValueError(f"Unexpected response format from TwelveLabs: {data}")
                    return segments[0]["float"]

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

        """
        return self.batch_size

    def get_tokenizer(self):
        """
        Instantiate and return the tokenizer used for preparing text for embedding.

        Marengo does not publish a public tokenizer; TikToken is used purely for
        chunk-size estimation, matching the FastembedEmbeddingEngine approach.

        Returns:
        --------

            A tokenizer object configured for the specified maximum token size.
        """
        logger.debug("Loading tokenizer for TwelveLabsEmbeddingEngine...")
        tokenizer = TikTokenTokenizer(
            model="gpt-4o", max_completion_tokens=self.max_completion_tokens
        )
        logger.debug("Tokenizer loaded for TwelveLabsEmbeddingEngine")
        return tokenizer
