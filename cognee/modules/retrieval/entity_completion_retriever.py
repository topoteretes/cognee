from typing import Any, Optional, List

from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.infrastructure.context.BaseContextProvider import BaseContextProvider
from cognee.tasks.entity_completion.entity_completion import entity_completion
from cognee.modules.retrieval.base_retriever import BaseRetriever


class EntityCompletionRetriever(BaseRetriever):
    """Retriever that uses entity-based completion for generating responses."""

    def __init__(
        self,
        extractor: BaseEntityExtractor,
        context_provider: BaseContextProvider,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.extractor = extractor
        self.context_provider = context_provider
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template

    async def get_context(self, query: str) -> Any:
        """Get context using entity extraction and context provider."""
        entities = await self.extractor.extract_entities(query)
        if not entities:
            return None
        return await self.context_provider.get_context(entities, query)

    async def get_completion(self, query: str, context: Optional[Any] = None) -> List[str]:
        """Generate completion using entity completion functionality."""
        result = await entity_completion(
            query=query,
            extractor=self.extractor,
            context_provider=self.context_provider,
            system_prompt_template=self.system_prompt_template,
            user_prompt_template=self.user_prompt_template,
        )
        return result
