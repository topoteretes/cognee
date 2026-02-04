import asyncio
from typing import Any, Optional, List, Type, Union
from cognee.shared.logging_utils import get_logger

from cognee.infrastructure.entities.BaseEntityExtractor import BaseEntityExtractor
from cognee.infrastructure.context.BaseContextProvider import BaseContextProvider
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion, summarize_text
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig


logger = get_logger("entity_completion_retriever")


class EntityCompletionRetriever(BaseRetriever):
    """
    Retriever that uses entity-based completion for generating responses.

    Public methods:

    - get_context
    - get_completion

    Instance variables:

    - extractor
    - context_provider
    - user_prompt_path
    - system_prompt_path
    """

    def __init__(
        self,
        extractor: BaseEntityExtractor,
        context_provider: BaseContextProvider,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        self.extractor = extractor
        self.context_provider = context_provider
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.session_id = session_id
        self.response_model = response_model

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Get relevant objects from the provided query.

        Extracts and returns entities from the provided query, returning None if no entities are found.

        Parameters:
        -----------

            - query (str): The query string for which context is being retrieved.

        Returns:
        --------

            - Any: The extracted entities, or None if no entities are found.

        """
        try:
            logger.info(f"Processing query: {query[:100]}")

            entities = await self.extractor.extract_entities(query)
            if not entities:
                logger.info("No entities extracted")
                return None

            return entities

        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return None

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Get context using the extracted entities and a context provider.

        Retrieves the context corresponding to the retrieved entities in retrieved_objects.
        Returns and empty string if no context is retrieved.

        Parameters:
        -----------

            - query (str): The query string for which context is being retrieved.
            - retrieved_objects (Any): The retrieved entities extracted from the query.

        Returns:
        --------

            - str: The context retrieved from the context provider or an empty string
            if not found or an error occurred.
        """
        try:
            logger.info(f"Processing query: {query[:100]}")

            context = await self.context_provider.get_context(retrieved_objects, query)
            if not context:
                logger.info("No context retrieved")
                return ""

            return context

        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return ""

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generate completion using provided context.

        Parameters:
        -----------

            - query (str): The query string for which completion is being generated.
            - retrieved_objects (Any): The retrieved objects extracted from the query.
            - context (Any): Optional context to be used for generating completion.

        Returns:
        --------

            - List[str]: A list containing the generated completion or an error message if no
              relevant entities were found.
        """
        try:
            # Check if we need to generate context summary for caching
            cache_config = CacheConfig()
            user = session_user.get()
            user_id = getattr(user, "id", None)
            session_save = user_id and cache_config.caching

            if session_save:
                conversation_history = await get_conversation_history(session_id=self.session_id)

                context_summary, completion = await asyncio.gather(
                    summarize_text(str(context)),
                    generate_completion(
                        query=query,
                        context=context,
                        user_prompt_path=self.user_prompt_path,
                        system_prompt_path=self.system_prompt_path,
                        conversation_history=conversation_history,
                        response_model=self.response_model,
                    ),
                )
            else:
                completion = await generate_completion(
                    query=query,
                    context=context,
                    user_prompt_path=self.user_prompt_path,
                    system_prompt_path=self.system_prompt_path,
                    response_model=self.response_model,
                )

            if session_save:
                await save_conversation_history(
                    query=query,
                    context_summary=context_summary,
                    answer=completion,
                    session_id=self.session_id,
                )

            return [completion]

        except Exception as e:
            logger.error(f"Completion generation failed: {str(e)}")
            return ["Completion generation failed"]
