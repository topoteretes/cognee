from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.tasks.completion.exceptions import NoRelevantDataFound


class CompletionRetriever(BaseRetriever):
    """Retriever for handling LLM-based completion searches."""

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path

    async def get_context(self, query: str) -> Any:
        """Retrieves relevant document chunks as context."""
        vector_engine = get_vector_engine()
        found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=1)
        if len(found_chunks) == 0:
            raise NoRelevantDataFound
        return found_chunks[0].payload["text"]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Generates an LLM completion using the context."""
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]
