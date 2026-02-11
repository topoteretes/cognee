from typing import Any, Optional, Type, List, Union

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig

logger = get_logger("TripletRetriever")


class TripletRetriever(BaseRetriever):
    """
    Retriever for handling LLM-based completion searches using triplets.

    Public methods:
    - get_context(query: str) -> str
    - get_completion(query: str, context: Optional[Any] = None) -> Any
    """

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.response_model = response_model

    async def get_retrieved_objects(self, query: str) -> Any:
        """
        Retrieves relevant triplets.

        Fetches triplets based on a query from a vector engine.
        Returns empty list if no triplets are found. Raises NoDataError if the collection is not
        found.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant triplets.

        Returns:
        --------

            - Any: A list containing the retrieved triplets, or an empty list if none are found.
        """
        vector_engine = get_vector_engine()

        try:
            if not await vector_engine.has_collection(collection_name="Triplet_text"):
                logger.error("Triplet_text collection not found")
                raise NoDataError(
                    "In order to use TRIPLET_COMPLETION first use the create_triplet_embeddings memify pipeline. "
                )

            found_triplets = await vector_engine.search(
                "Triplet_text", query, limit=self.top_k, include_payload=True
            )

            if len(found_triplets) == 0:
                return []

            return found_triplets
        except CollectionNotFoundError as error:
            logger.error("Triplet_text collection not found")
            raise NoDataError("No data found in the system, please add data first.") from error

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        if retrieved_objects:
            triplets_payload = [
                found_triplet.payload["text"] for found_triplet in retrieved_objects
            ]
            combined_context = "\n".join(triplets_payload)
            return combined_context
        return ""

    def _completion_kwargs(self, context: str) -> dict:
        """Common kwargs for completion calls (no session)."""
        return {
            "context": context,
            "user_prompt_path": self.user_prompt_path,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt": self.system_prompt,
            "response_model": self.response_model,
        }

    async def _generate_completion_without_session(self, query: str, context: str) -> List[Any]:
        """Generate completion without session; returns list of one completion."""
        kwargs = self._completion_kwargs(context)
        completion = await generate_completion(query=query, **kwargs)
        return [completion]

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
        """
        Generates an LLM completion using the context.

        Retrieves context if not provided and generates a completion based on the query and
        context using an external completion generator.

        Parameters:
        -----------

            - query (str): The query string to be used for generating a completion.
            - context (Optional[Any]): Optional pre-fetched context to use for generating the
              completion; if None, it retrieves the context for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)
            - response_model (Type): The Pydantic model type for structured output. (default str)

        Returns:
        --------

            - Any: The generated completion based on the provided query and context.
        """
        cache_config = CacheConfig()
        user = session_user.get()
        user_id = getattr(user, "id", None)
        use_session = user_id and cache_config.caching

        if use_session:
            sm = get_session_manager()
            completion = await sm.run_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
                summarize_context=False,
            )
            return [completion]
        return await self._generate_completion_without_session(query, context)
