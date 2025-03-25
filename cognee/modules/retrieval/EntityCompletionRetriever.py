from typing import Any, Optional, List
from cognee.shared.logging_utils import get_logger

from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.infrastructure.context.BaseContextProvider import BaseContextProvider
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion


logger = get_logger("entity_completion_retriever")


class EntityCompletionRetriever(BaseRetriever):
    """Retriever that uses entity-based completion for generating responses."""

    def __init__(
        self,
        extractor: BaseEntityExtractor,
        context_provider: BaseContextProvider,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
    ):
        self.extractor = extractor
        self.context_provider = context_provider
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path

    async def get_context(self, query: str) -> Any:
        """Get context using entity extraction and context provider."""
        try:
            logger.info(f"Processing query: {query[:100]}")

            entities = await self.extractor.extract_entities(query)
            if not entities:
                logger.info("No entities extracted")
                return None

            context = await self.context_provider.get_context(entities, query)
            if not context:
                logger.info("No context retrieved")
                return None

            return context

        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return None

    async def get_completion(self, query: str, context: Optional[Any] = None) -> List[str]:
        """Generate completion using provided context or fetch new context."""
        try:
            if context is None:
                context = await self.get_context(query)

            if context is None:
                return ["No relevant entities found for the query."]

            completion = await generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
            )
            return [completion]

        except Exception as e:
            logger.error(f"Completion generation failed: {str(e)}")
            return ["Completion generation failed"]
