import asyncio

# Remove the broken 'from typing import isinstance' line
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
)


# 1. Create a dummy class that implements the Protocol properly
class ValidMockEngine:
    async def embed_text(self, text: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]

    def get_vector_size(self) -> int:
        return 3

    def get_batch_size(self) -> int:
        return 32


# 2. Create a broken dummy class missing a method
class InvalidMockEngine:
    def get_vector_size(self) -> int:
        return 3


async def run_tests():
    valid_instance = ValidMockEngine()
    invalid_instance = InvalidMockEngine()

    # Test A: Check structural subtyping validation via runtime_checkable
    print("--- Running Protocol Checks ---")
    print(
        "Does ValidMockEngine match protocol?:",
        isinstance(valid_instance, EmbeddingEngine),
    )  # Should be True
    print(
        "Does InvalidMockEngine match protocol?:",
        isinstance(invalid_instance, EmbeddingEngine),
    )  # Should be False

    # Test B: Make sure executing methods on the valid instance works flawlessly
    result = await valid_instance.embed_text(["hello"])
    print("Method execution check (Embed output):", result)


if __name__ == "__main__":
    asyncio.run(run_tests())
