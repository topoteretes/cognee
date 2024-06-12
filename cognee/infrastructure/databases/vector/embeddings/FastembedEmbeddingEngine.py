from typing import List, Optional
from fastembed import TextEmbedding
from cognee.root_dir import get_absolute_path
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine

class FastembedEmbeddingEngine(EmbeddingEngine):
    embedding_model: str
    embedding_dimensions: int

    def __init__(
        self,
        embedding_model: Optional[str] = "BAAI/bge-large-en-v1.5",
        embedding_dimensions: Optional[int] = 1024,
    ):
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def embed_text(self, text: List[str]) -> List[float]:
        embedding_model = TextEmbedding(model_name = self.embedding_model, cache_dir = get_absolute_path("cache/embeddings"))
        embeddings_list = list(map(lambda embedding: embedding.tolist(), embedding_model.embed(text)))

        return embeddings_list

    def get_vector_size(self) -> int:
        return self.embedding_dimensions
