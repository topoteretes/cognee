import asyncio
from typing import List

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.shared.rate_limiting import embedding_rate_limiter_context_manager


class MockEmbeddingEngine(LiteLLMEmbeddingEngine):
    """
    Mock version of LiteLLMEmbeddingEngine that returns fixed embeddings
    and can be configured to simulate rate limiting and failures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock = True
        self.fail_every_n_requests = 0
        self.request_count = 0
        self.add_delay = 0

    def configure_mock(self, fail_every_n_requests=0, add_delay=0):
        """
        Configure the mock's behavior

        Args:
            fail_every_n_requests: Raise an exception every n requests (0 = never fail)
            add_delay: Add artificial delay in seconds to each request
        """
        self.fail_every_n_requests = fail_every_n_requests
        self.add_delay = add_delay

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Mock implementation that returns fixed embeddings and can
        simulate failures and delays based on configuration.
        """
        self.request_count += 1

        # Simulate processing delay if configured
        if self.add_delay > 0:
            await asyncio.sleep(self.add_delay)

        # Simulate failures if configured
        if self.fail_every_n_requests > 0 and self.request_count % self.fail_every_n_requests == 0:
            raise Exception(f"Mock failure on request #{self.request_count}")

        # Return mock embeddings of the correct dimension
        async with embedding_rate_limiter_context_manager():
            return [[0.1] * self.dimensions for _ in text]
