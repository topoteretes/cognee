from typing import List, Protocol

class EmbeddingEngine(Protocol):
    async def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError()

    def get_vector_size(self) -> int:
        raise NotImplementedError()
