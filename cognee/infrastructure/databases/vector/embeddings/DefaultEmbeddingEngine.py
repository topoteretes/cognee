from typing import List

import instructor
from openai import AsyncOpenAI
from fastembed import TextEmbedding
from fastembed import TextEmbedding
from openai import AsyncOpenAI

from cognee.config import Config
from cognee.root_dir import get_absolute_path
from .EmbeddingEngine import EmbeddingEngine

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

        OPENAI_API_KEY = config.openai_key

        aclient = instructor.patch(AsyncOpenAI())
        text = text.replace("\n", " ")
        response = await aclient.embeddings.create(input = text, model = config.openai_embedding_model)
        embedding = response.data[0].embedding
        # embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))
        return embedding


    def get_vector_size(self) -> int:
        return config.openai_embedding_dimensions

