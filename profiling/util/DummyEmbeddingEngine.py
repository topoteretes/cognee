import numpy as np
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine

class DummyEmbeddingEngine(EmbeddingEngine):
    async def embed_text(self, text: list[str]) -> list[list[float]]:
        return(list(list(np.random.randn(3072))))

    def get_vector_size(self) -> int:
        return(3072)
