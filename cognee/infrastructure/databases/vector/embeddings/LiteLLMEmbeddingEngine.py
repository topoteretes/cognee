import asyncio
import logging
import math
from typing import List, Optional
import litellm
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine

litellm.set_verbose = False
logger = logging.getLogger("LiteLLMEmbeddingEngine")

class LiteLLMEmbeddingEngine(EmbeddingEngine):
    api_key: str
    endpoint: str
    api_version: str
    model: str
    dimensions: int

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

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        try:
            response = await litellm.aembedding(
                self.model,
                input = text,
                api_key = self.api_key,
                api_base = self.endpoint,
                api_version = self.api_version
            )
            return [data["embedding"] for data in response.data]

        except litellm.exceptions.ContextWindowExceededError as error:
            if isinstance(text, list):
                parts = [text[0:math.ceil(len(text)/2)], text[math.ceil(len(text)/2):]]
                parts_futures = [self.embed_text(part) for part in parts]
                embeddings = await asyncio.gather(*parts_futures)

                all_embeddings = []
                for embeddings_part in embeddings:
                    all_embeddings.extend(embeddings_part)

                return [data["embedding"] for data in all_embeddings]

            logger.error("Context window exceeded for embedding text: %s", str(error))
            raise error

        except Exception as error:
            logger.error("Error embedding text: %s", str(error))
            raise error

    def get_vector_size(self) -> int:
        return self.dimensions
