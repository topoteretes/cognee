import asyncio
from typing import List, Optional
import litellm
from litellm import aembedding
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine

litellm.set_verbose = False

class LiteLLMEmbeddingEngine(EmbeddingEngine):
    api_key: str
    embedding_model: str
    embedding_dimensions: int

    def __init__(
        self,
        embedding_model: Optional[str] = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = 3072,
        api_key: str = None,
    ):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        async def get_embedding(text_):
            response = await aembedding(
                self.embedding_model,
                input = text_,
                api_key = self.api_key
            )

            return response.data[0]["embedding"]

        tasks = [get_embedding(text_) for text_ in text]
        result = await asyncio.gather(*tasks)
        return result

    def get_vector_size(self) -> int:
        return self.embedding_dimensions
