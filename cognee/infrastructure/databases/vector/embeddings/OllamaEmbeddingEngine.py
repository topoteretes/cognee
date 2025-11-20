import asyncio
from cognee.shared.logging_utils import get_logger
import aiohttp
from typing import List, Optional
import os
import litellm
import logging
import aiohttp.http_exceptions
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential_jitter,
    retry_if_not_exception_type,
    before_sleep_log,
)

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.llm.tokenizer.HuggingFace import (
    HuggingFaceTokenizer,
)
from cognee.infrastructure.databases.vector.embeddings.embedding_rate_limiter import (
    embedding_rate_limit_async,
    embedding_sleep_and_retry_async,
)
from cognee.shared.utils import create_secure_ssl_context

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
        endpoint: Optional[str] = "http://localhost:11434/api/embeddings",
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
        if self.mock:
            return [[0.0] * self.dimensions for _ in text]

        embeddings = await asyncio.gather(*[self._get_embedding(prompt) for prompt in text])
        return embeddings

    @retry(
        stop=stop_after_delay(128),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(litellm.exceptions.NotFoundError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    async def _get_embedding(self, prompt: str) -> List[float]:
        """
        Internal method to call the Ollama embeddings endpoint for a single prompt.
        """
        payload = {"model": self.model, "prompt": prompt, "input": prompt}

        headers = {}
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self.endpoint, json=payload, headers=headers, timeout=60.0
            ) as response:
                data = await response.json()
                if "embeddings" in data:
                    return data["embeddings"][0]
                else:
                    return data["data"][0]["embedding"]

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
