import asyncio
from typing import List

import instructor
from openai import AsyncOpenAI
from fastembed import TextEmbedding

from cognee.config import Config
from cognee.root_dir import get_absolute_path
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from litellm import aembedding
import litellm

litellm.set_verbose = True
from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
config = get_embedding_config()

class DefaultEmbeddingEngine(EmbeddingEngine):
    async def embed_text(self, text: List[str]) -> List[float]:
        embedding_model = TextEmbedding(model_name = config.embedding_model, cache_dir = get_absolute_path("cache/embeddings"))
        embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))

        return embeddings_list

    def get_vector_size(self) -> int:
        return config.embedding_dimensions


class LiteLLMEmbeddingEngine(EmbeddingEngine):
    import asyncio
    from typing import List

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        async def get_embedding(text_):
            response = await aembedding(config.litellm_embedding_model, input=text_)
            return response.data[0]['embedding']

        tasks = [get_embedding(text_) for text_ in text]
        result = await asyncio.gather(*tasks)
        return result

        # embedding = response.data[0].embedding
        # # embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))
        # print("response", type(response.data[0]['embedding']))
        # print("response", response.data[0])
        # return [response.data[0]['embedding']]


    def get_vector_size(self) -> int:
        return config.litellm_embedding_dimensions


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

