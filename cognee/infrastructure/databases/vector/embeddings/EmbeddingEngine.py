from typing import Protocol


class EmbeddingEngine(Protocol):
    async def embed_text(self, text: list[str]) -> list[list[float]]:
        raise NotImplementedError()

    def get_vector_size(self) -> int:
        raise NotImplementedError()
