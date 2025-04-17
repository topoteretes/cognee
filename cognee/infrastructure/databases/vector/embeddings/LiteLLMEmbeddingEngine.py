import asyncio
from cognee.shared.logging_utils import get_logger
from typing import List, Optional
import numpy as np
import math
import litellm
import os
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.Gemini import GeminiTokenizer
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
from cognee.infrastructure.llm.tokenizer.Mistral import MistralTokenizer
from cognee.infrastructure.llm.tokenizer.TikToken import TikTokenTokenizer
from cognee.infrastructure.llm.embedding_rate_limiter import (
    embedding_rate_limit_async,
    embedding_sleep_and_retry_async,
)

litellm.set_verbose = False
logger = get_logger("LiteLLMEmbeddingEngine")


class LiteLLMEmbeddingEngine(EmbeddingEngine):
    api_key: str
    endpoint: str
    api_version: str
    provider: str
    model: str
    dimensions: int
    mock: bool

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = "openai/text-embedding-3-large",
        provider: str = "openai",
        dimensions: Optional[int] = 3072,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
        max_tokens: int = 512,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.tokenizer = self.get_tokenizer()
        self.retry_count = 0

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    @embedding_sleep_and_retry_async()
    @embedding_rate_limit_async
    async def embed_text(self, text: List[str]) -> List[List[float]]:
        try:
            if self.mock:
                response = {"data": [{"embedding": [0.0] * self.dimensions} for _ in text]}
                return [data["embedding"] for data in response["data"]]
            else:
                response = await litellm.aembedding(
                    model=self.model,
                    input=text,
                    api_key=self.api_key,
                    api_base=self.endpoint,
                    api_version=self.api_version,
                )

                return [data["embedding"] for data in response.data]

        except litellm.exceptions.ContextWindowExceededError as error:
            if isinstance(text, list) and len(text) > 1:
                mid = math.ceil(len(text) / 2)
                left, right = text[:mid], text[mid:]
                left_vecs, right_vecs = await asyncio.gather(
                    self.embed_text(left),
                    self.embed_text(right),
                )
                return left_vecs + right_vecs

                # If caller passed ONE oversize string split the string itself into
                # half so we can process it
            if isinstance(text, list) and len(text) == 1:
                logger.debug(f"Pooling embeddings of text string with size: {len(text[0])}")
                s = text[0]
                third = len(s) // 3
                # We are using thirds to intentionally have overlap between split parts
                # for better embedding calculation
                left_part, right_part = s[: third * 2], s[third:]

                # Recursively embed the split parts in parallel
                (left_vec,), (right_vec,) = await asyncio.gather(
                    self.embed_text([left_part]),
                    self.embed_text([right_part]),
                )

                # POOL the two embeddings into one
                pooled = (np.array(left_vec) + np.array(right_vec)) / 2
                return [pooled.tolist()]

            logger.error("Context window exceeded for embedding text: %s", str(error))
            raise error

        except (
            litellm.exceptions.BadRequestError,
            litellm.exceptions.NotFoundError,
        ) as e:
            logger.error(f"Embedding error with model {self.model}: {str(e)}")
            raise EmbeddingException(f"Failed to index data points using model {self.model}")

        except Exception as error:
            logger.error("Error embedding text: %s", str(error))
            raise error

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_tokenizer(self):
        logger.debug(f"Loading tokenizer for model {self.model}...")
        # If model also contains provider information, extract only model information
        model = self.model.split("/")[-1]

        if "openai" in self.provider.lower():
            tokenizer = TikTokenTokenizer(model=model, max_tokens=self.max_tokens)
        elif "gemini" in self.provider.lower():
            tokenizer = GeminiTokenizer(model=model, max_tokens=self.max_tokens)
        elif "mistral" in self.provider.lower():
            tokenizer = MistralTokenizer(model=model, max_tokens=self.max_tokens)
        else:
            tokenizer = HuggingFaceTokenizer(model=self.model, max_tokens=self.max_tokens)

        logger.debug(f"Tokenizer loaded for model: {self.model}")
        return tokenizer
