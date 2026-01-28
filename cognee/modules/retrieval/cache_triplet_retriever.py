"""
Triplet-style retriever that reads from the cache vector DB (Redis) instead of the default vector DB.

Uses the same completion flow as TripletRetriever but get_context() queries the "cache"
collection via get_cache_vector_engine(). Works with cognee.search when query_type is
SearchType.TRIPLET_COMPLETION_CACHE.
"""

import asyncio
from typing import Any, Optional, Type, List

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_cache_vector_engine
from cognee.modules.retrieval.utils.completion import generate_completion, summarize_text
from cognee.modules.retrieval.utils.session_cache import (
    save_conversation_history,
    get_conversation_history,
)
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig

logger = get_logger("CacheTripletRetriever")


# Must match the collection used by index_datapoints_into_cache
CACHE_COLLECTION_NAME = "cache"


def _get_text_from_payload(payload: Optional[dict]) -> str:
    """Extract display text from a cache result payload (full DataPoint model_dump)."""
    if not payload:
        return ""
    if "text" in payload:
        return str(payload["text"])
    meta = payload.get("metadata") or {}
    index_fields = meta.get("index_fields") or ["text"]
    if index_fields:
        return str(payload.get(index_fields[0], ""))
    return ""


class CacheTripletRetriever(BaseRetriever):
    """
    Same interface as TripletRetriever but retrieves context from the cache vector DB (Redis)
    collection "cache" instead of the default vector DB collection "Triplet_text".

    Use with cognee.search(query_type=SearchType.TRIPLET_COMPLETION_CACHE, ...).
    """

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
    ):
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.system_prompt = system_prompt

    async def get_context(self, query: str) -> str:
        """
        Retrieves relevant items from the cache vector DB as context.

        Queries the "cache" collection via the cache vector engine (Redis).
        Returns empty string if no results. Raises NoDataError if the collection
        does not exist.
        """
        cache_engine = get_cache_vector_engine()

        try:
            if not await cache_engine.has_collection(collection_name=CACHE_COLLECTION_NAME):
                logger.error("Cache collection not found")
                raise NoDataError(
                    "Cache vector collection not found. Use index_datapoints_into_cache to populate the cache first."
                )

            found = await cache_engine.search(
                CACHE_COLLECTION_NAME, query, limit=self.top_k, include_payload=True
            )

            if len(found) == 0:
                return ""

            texts = [_get_text_from_payload(f.payload) for f in found]
            combined_context = "\n".join(texts)
            return combined_context
        except CollectionNotFoundError as error:
            logger.error("Cache collection not found")
            raise NoDataError(
                "Cache vector collection not found. Use index_datapoints_into_cache to populate the cache first."
            ) from error

    async def get_completion(
        self,
        query: str,
        context: Optional[Any] = None,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ) -> List[Any]:
        """Same completion flow as TripletRetriever."""
        if context is None:
            context = await self.get_context(query)

        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        session_save = user_id and cache_config.caching

        if session_save:
            completion = await self._get_completion_with_session(
                query=query,
                context=context,
                session_id=session_id,
                response_model=response_model,
            )
        else:
            completion = await self._get_completion_without_session(
                query=query,
                context=context,
                response_model=response_model,
            )

        return [completion]

    async def _get_completion_with_session(
        self,
        query: str,
        context: str,
        session_id: Optional[str],
        response_model: Type,
    ) -> Any:
        """Generate completion with session history and caching."""
        conversation_history = await get_conversation_history(session_id=session_id)

        context_summary, completion = await asyncio.gather(
            summarize_text(context),
            generate_completion(
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                conversation_history=conversation_history,
                response_model=response_model,
            ),
        )

        await save_conversation_history(
            query=query,
            context_summary=context_summary,
            answer=completion,
            session_id=session_id,
        )

        return completion

    async def _get_completion_without_session(
        self,
        query: str,
        context: str,
        response_model: Type,
    ) -> Any:
        """Generate completion without session history."""
        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=response_model,
        )

        return completion
