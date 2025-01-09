import asyncio
import logging
import math
from typing import List, Optional
import litellm
import os
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException

litellm.set_verbose = False
logger = logging.getLogger("LiteLLMEmbeddingEngine")


class LiteLLMEmbeddingEngine(EmbeddingEngine):
    api_key: str
    endpoint: str
    api_version: str
    model: str
    dimensions: int
    mock: bool

    def __init__(
        self,
        model: Optional[str] = "text-embedding-3-large",
        dimensions: Optional[int] = 3072,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model
        self.dimensions = dimensions

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    MAX_RETRIES = 5
    retry_count = 0

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        async def exponential_backoff(attempt):
            wait_time = min(10 * (2**attempt), 60)  # Max 60 seconds
            await asyncio.sleep(wait_time)

        try:
            if self.mock:
                response = {"data": [{"embedding": [0.0] * self.dimensions} for _ in text]}

                self.retry_count = 0

                return [data["embedding"] for data in response["data"]]
            else:
                response = await litellm.aembedding(
                    self.model,
                    input=text,
                    api_key=self.api_key,
                    api_base=self.endpoint,
                    api_version=self.api_version,
                )

                self.retry_count = 0

                return [data["embedding"] for data in response.data]

        except litellm.exceptions.ContextWindowExceededError as error:
            if isinstance(text, list):
                if len(text) == 1:
                    parts = [text]
                else:
                    parts = [text[0 : math.ceil(len(text) / 2)], text[math.ceil(len(text) / 2) :]]

                parts_futures = [self.embed_text(part) for part in parts]
                embeddings = await asyncio.gather(*parts_futures)

                all_embeddings = []
                for embeddings_part in embeddings:
                    all_embeddings.extend(embeddings_part)

                return all_embeddings

            logger.error("Context window exceeded for embedding text: %s", str(error))
            raise error

        except litellm.exceptions.RateLimitError:
            if self.retry_count >= self.MAX_RETRIES:
                raise Exception("Rate limit exceeded and no more retries left.")

            await exponential_backoff(self.retry_count)

            self.retry_count += 1

            return await self.embed_text(text)

        except (litellm.exceptions.BadRequestError, litellm.llms.OpenAI.openai.OpenAIError):
            raise EmbeddingException("Failed to index data points.")

        except Exception as error:
            logger.error("Error embedding text: %s", str(error))
            raise error

    def get_vector_size(self) -> int:
        return self.dimensions
