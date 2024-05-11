import asyncio
from typing import List

import instructor
from openai import AsyncOpenAI
from fastembed import TextEmbedding
from fastembed import TextEmbedding
from openai import AsyncOpenAI

from cognee.config import Config
from cognee.root_dir import get_absolute_path
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from litellm import aembedding
config = Config()
config.load()

class DefaultEmbeddingEngine(EmbeddingEngine):
    async def embed_text(self, text: List[str]) -> List[float]:
        embedding_model = TextEmbedding(model_name = config.embedding_model, cache_dir = get_absolute_path("cache/embeddings"))
        embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))

        return embeddings_list

    def get_vector_size(self) -> int:
        return config.embedding_dimensions


class OpenAIEmbeddingEngine(EmbeddingEngine):
    async def embed_text(self, text: List[str]) -> List[float]:


        response = await aembedding(config.openai_embedding_model, input=text)


        # embedding = response.data[0].embedding
        # embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))
        return response.data[0]['embedding']


    def get_vector_size(self) -> int:
        return config.openai_embedding_dimensions


if __name__ == "__main__":
    async def gg():
        openai_embedding_engine = OpenAIEmbeddingEngine()
        # print(openai_embedding_engine.embed_text(["Hello, how are you?"]))
        # print(openai_embedding_engine.get_vector_size())
        # default_embedding_engine = DefaultEmbeddingEngine()
        sds = await openai_embedding_engine.embed_text(["Hello, sadasdas are you?"])
        print(sds)
        # print(default_embedding_engine.get_vector_size())

    asyncio.run(gg())

