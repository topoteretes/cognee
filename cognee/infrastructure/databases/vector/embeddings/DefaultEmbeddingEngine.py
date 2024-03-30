from typing import List
from fastembed import TextEmbedding
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
