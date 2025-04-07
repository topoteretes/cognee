import asyncio
import os
from typing import List, Optional

import openai
from openai import (
    APIError,
    RateLimitError,
    APIConnectionError,
)

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.TikToken.adapter import TikTokenTokenizer
from cognee.infrastructure.llm.rate_limiter import (
    embedding_rate_limit_async,
    embedding_sleep_and_retry_async
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("OpenAIEmbeddingEngine")


class OpenAIEmbeddingEngine(EmbeddingEngine):
    model: str
    dimensions: int
    max_tokens: int
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self, model: Optional[str] = None, dimensions: Optional[int] = None, max_tokens: int = 8191,
    ):
        # Azure specific env vars
        self.api_type = os.getenv("OPENAI_API_TYPE", "open_ai")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") or model or "text-embedding-ada-002"

        # Use Azure model name to compute embedding dimensions if not provided
        if dimensions is None:
            if "text-embedding-3-small" in self.embedding_model:
                dimensions = 1536
            elif "text-embedding-3-large" in self.embedding_model:
                dimensions = 3072
            elif "text-embedding-ada-002" in self.embedding_model:
                dimensions = 1536
            else:
                dimensions = 1536  # Default to Ada dimensions for unknown models

        self.model = self.embedding_model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    @embedding_sleep_and_retry_async()
    @embedding_rate_limit_async
    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Given a list of text prompts, returns a list of embedding vectors.
        """
        if self.mock:
            return [[0.0] * self.dimensions for _ in text]

        api_key = os.getenv("OPENAI_API_KEY")
        api_organization = os.getenv("OPENAI_ORGANIZATION")
        
        client_kwargs = {"api_key": api_key}
        if api_organization:
            client_kwargs["organization"] = api_organization
        if self.api_type.lower() == "azure":
            client_kwargs["azure_endpoint"] = self.api_base
            client_kwargs["api_version"] = os.getenv("OPENAI_API_VERSION", "2023-05-15")
        else:
            client_kwargs["base_url"] = self.api_base

        client = openai.AsyncOpenAI(**client_kwargs)

        if self.api_type.lower() == "azure":
            model = self.embedding_deployment
        else:
            model = self.model

        response = await client.embeddings.create(input=text, model=model)
        return [embed.embedding for embed in response.data]

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_tokenizer(self):
        logger.debug("Loading Tiktoken tokenizer for OpenAIEmbeddingEngine...")
        tokenizer = TikTokenTokenizer(self.model, max_tokens=self.max_tokens)
        logger.debug("Tokenizer loaded for OpenAIEmbeddingEngine")
        return tokenizer 