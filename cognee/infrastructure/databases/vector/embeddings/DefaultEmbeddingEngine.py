import asyncio
from typing import List, Optional
from openai import AsyncOpenAI
from fastembed import TextEmbedding

from cognee.root_dir import get_absolute_path
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from litellm import aembedding
import litellm

litellm.set_verbose = True

class DefaultEmbeddingEngine(EmbeddingEngine):
    embedding_model: str
    embedding_dimensions: int
    def __init__(
        self,
        embedding_model: Optional[str],
        embedding_dimensions: Optional[int],
    ):
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def embed_text(self, text: List[str]) -> List[float]:
        embedding_model = TextEmbedding(model_name = self.embedding_model, cache_dir = get_absolute_path("cache/embeddings"))
        embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))

        return embeddings_list

    def get_vector_size(self) -> int:
        return self.embedding_dimensions


class LiteLLMEmbeddingEngine(EmbeddingEngine):
    embedding_model: str
    embedding_dimensions: int
    def __init__(
        self,
        embedding_model: Optional[str],
        embedding_dimensions: Optional[int],
    ):
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
    import asyncio
    from typing import List

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        async def get_embedding(text_):
            response = await aembedding(self.embedding_model, input=text_)
            return response.data[0]['embedding']

        tasks = [get_embedding(text_) for text_ in text]
        result = await asyncio.gather(*tasks)
        return result
    def get_vector_size(self) -> int:
        return self.embedding_dimensions


if __name__ == "__main__":
    async def gg():
        openai_embedding_engine = LiteLLMEmbeddingEngine()
        # print(openai_embedding_engine.embed_text(["Hello, how are you?"]))
        # print(openai_embedding_engine.get_vector_size())
        # default_embedding_engine = DefaultEmbeddingEngine()
        sds = await openai_embedding_engine.embed_text(["Hello, sadasdas are you?"])
        print(sds)
        # print(default_embedding_engine.get_vector_size())

    asyncio.run(gg())

