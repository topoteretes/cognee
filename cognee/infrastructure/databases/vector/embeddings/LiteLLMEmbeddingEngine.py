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
    MAX_RETRIES = 3
    
    PROVIDER_CONFIGS = {
        "openai": {
            "model": "text-embedding-3-large",
            "dimensions": 3072,
            "api_base": "https://api.openai.com/v1"
        },
        "gemini": {
            "model": "text-embedding-004",
            "dimensions": 768,
            "api_base": "https://generativelanguage.googleapis.com/v1beta"
        }
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
    ):
        self.provider = provider.lower()
        provider_config = self.PROVIDER_CONFIGS.get(self.provider)
        if not provider_config:
            raise ValueError(f"Unsupported provider: {provider}")

        self.model = model or provider_config["model"]
        self.dimensions = dimensions or provider_config["dimensions"]
        self.api_key = api_key
        self.endpoint = endpoint or provider_config["api_base"]
        self.api_version = api_version
        self.retry_count = 0

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def exponential_backoff(self, attempt: int) -> None:
        wait_time = min(10 * (2 ** attempt), 60)  # Max 60 seconds
        await asyncio.sleep(wait_time)

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        try:
            if self.mock:
                response = {
                    "data": [{"embedding": [0.0] * self.dimensions} for _ in text]
                }
                return [data["embedding"] for data in response["data"]]
            else:
                # Configure model name based on provider
                if self.provider == "gemini":
                    model_name = f"gemini/{self.model}"
                    # For Gemini, we need to ensure we're using their specific endpoint format
                    api_base = f"{self.endpoint}/models/{self.model}:embedContent"
                else:
                    model_name = self.model
                    api_base = self.endpoint

                response = await litellm.aembedding(
                    model=model_name,
                    input=text,
                    api_key=self.api_key,
                    api_base=api_base,
                    api_version=self.api_version
                )

                self.retry_count = 0  # Reset retry count on successful call
                return [data["embedding"] for data in response.data]

        except litellm.exceptions.ContextWindowExceededError as error:
            if isinstance(text, list):
                if len(text) == 1:
                    parts = [text]
                else:
                    parts = [text[0:math.ceil(len(text) / 2)], text[math.ceil(len(text) / 2):]]

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
                raise Exception(f"Rate limit exceeded and no more retries left.")

            await self.exponential_backoff(self.retry_count)
            self.retry_count += 1
            return await self.embed_text(text)

        except (litellm.exceptions.BadRequestError, 
                litellm.exceptions.NotFoundError,
                litellm.llms.OpenAI.openai.OpenAIError) as e:
            logger.error(f"Embedding error with provider {self.provider}: {str(e)}")
            raise EmbeddingException(f"Failed to index data points using {self.provider} provider with model {self.model}")

        except Exception as error:
            logger.error("Error embedding text: %s", str(error))
            raise error

    def get_vector_size(self) -> int:
        return self.dimensions