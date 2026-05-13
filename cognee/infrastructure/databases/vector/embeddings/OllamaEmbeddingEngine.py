import asyncio
import math
from cognee.shared.logging_utils import get_logger
import aiohttp
from typing import List, Optional
import os
import litellm
import logging
import aiohttp.http_exceptions
import numpy as np
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.llm.tokenizer.HuggingFace import (
    HuggingFaceTokenizer,
)
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager
from cognee.shared.utils import create_secure_ssl_context
from cognee.infrastructure.databases.vector.embeddings.utils import (
    sanitize_embedding_text_inputs,
    handle_embedding_response,
)

logger = get_logger("OllamaEmbeddingEngine")


class OllamaEmbeddingEngine(EmbeddingEngine):
    """
    Implements an embedding engine using the Ollama embedding model.

    Public methods:
    - embed_text
    - get_vector_size
    - get_tokenizer

    Instance variables:
    - model
    - dimensions
    - max_completion_tokens
    - endpoint
    - mock
    - huggingface_tokenizer_name
    - tokenizer
    """

    model: str
    dimensions: int
    max_completion_tokens: int
    endpoint: str
    mock: bool
    huggingface_tokenizer_name: str

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = "avr/sfr-embedding-mistral:latest",
        dimensions: Optional[int] = 1024,
        max_completion_tokens: int = 512,
        endpoint: Optional[str] = "http://localhost:11434/api/embed",
        huggingface_tokenizer: str = "Salesforce/SFR-Embedding-Mistral",
        batch_size: int = 100,
    ):
        self.model = model
        self.dimensions = dimensions
        self.max_completion_tokens = max_completion_tokens
        self.endpoint = endpoint
        self.huggingface_tokenizer_name = huggingface_tokenizer
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of text prompts.

        If mocking is enabled, returns a list of zero vectors instead of actual embeddings.

        Parameters:
        -----------

            - text (List[str]): A list of text prompts for which to generate embeddings.

        Returns:
        --------

            - List[List[float]]: A list of embedding vectors corresponding to the text prompts.
        """
        original_texts = text if isinstance(text, list) else [text]
        sanitized_text = sanitize_embedding_text_inputs(original_texts)

        if self.mock:
            embeddings = [[0.0] * self.dimensions for _ in sanitized_text]
            return handle_embedding_response(original_texts, embeddings, self.dimensions)

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

            logger.error(f"Embedding error in OllamaEmbeddingEngine: {str(error)}")
            raise EmbeddingException(
                f"Failed to index data points using model {self.model}"
            ) from error

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(8, 128),
        retry=retry_if_not_exception_type((litellm.exceptions.NotFoundError, ValueError)),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def _get_embedding(self, prompt: str) -> List[float]:
        """
        Internal method to call the Ollama embeddings endpoint for a single prompt.
        """

        payload = {
            "model": self.model,
            "input": prompt,
            "dimensions": self.dimensions,
        }

        headers = {}
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with embedding_rate_limiter_context_manager():
                async with session.post(
                    self.endpoint, json=payload, headers=headers, timeout=60.0
                ) as response:
                    data = await response.json()

                    if "error" in data:
                        error_msg = data["error"]
                        logger.error(f"Ollama embedding error: {error_msg}")
                        if "context length" in error_msg or "input length" in error_msg:
                            raise ValueError(f"Text too long for embedding model: {error_msg}")
                        raise RuntimeError(f"Ollama embedding API error: {error_msg}")

                    if "embeddings" in data:
                        return data["embeddings"][0]
                    elif "embedding" in data:
                        return data["embedding"]
                    elif "data" in data and len(data["data"]) > 0:
                        return data["data"][0]["embedding"]
                    else:
                        raise ValueError(f"Unexpected response format from Ollama: {data}")

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
        Return the desired batch size for embedding calls

        Returns:

        """
        return self.batch_size

    def get_tokenizer(self):
        """
        Load and return a HuggingFace tokenizer for the embedding engine.

        Returns:
        --------

            The instantiated HuggingFace tokenizer used by the embedding engine.
        """
        logger.debug("Loading HuggingfaceTokenizer for OllamaEmbeddingEngine...")
        tokenizer = HuggingFaceTokenizer(
            model=self.huggingface_tokenizer_name, max_completion_tokens=self.max_completion_tokens
        )
        logger.debug("Tokenizer loaded for OllamaEmbeddingEngine")
        return tokenizer
