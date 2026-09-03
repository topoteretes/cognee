from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from cognee.modules.search.models.EvidenceReference import EvidenceReference


class BaseRetriever(ABC):
    """
    Base class for all retrieval operations.

    The retrieval workflow follows a three-step pipeline:
    1. get_retrieved_objects: Fetch raw data (e.g., Graph Edges, Vector chunks).
    2. get_context_from_objects: Process raw data into a format suitable for an LLM.
    3. get_completion_from_context: Generate a final response with the help of an LLM
       using the context and original query.
    """

    # Deterministic retrievers can opt out of conversational session analysis.
    # That analysis may call an LLM before retrieval, which is not appropriate
    # for search types whose contract is explicitly non-generative.
    supports_session_turn_preparation = True

    # Whether get_completion_from_context sends exactly one prompt built from this
    # retriever's (user_prompt_path, system_prompt_path). only_context previews render
    # that pair; retrievers that never prompt an LLM, or that run several rounds on
    # other templates, opt out so a preview does not invent a prompt for them.
    supports_prompt_preview = True

    @abstractmethod
    async def get_retrieved_objects(self, query: Optional[str], query_batch: Optional[str]) -> Any:
        """
        Retrieves the raw data points from the underlying storage (Graph or Vector DB).

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.

        Returns:
            List[Any]: A list of raw objects (e.g., Edge objects, Document chunks)
                       relevant to the query.
        """
        pass

    @abstractmethod
    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[str] = None,
        retrieved_objects: Any = None,
    ) -> Union[str, List[str]]:
        """
        Transforms raw retrieved objects into a structured context for the LLM.

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.

        Returns:
            Any: The formatted context (typically a string or a list of strings)
                 to be injected into a prompt.
        """
        pass

    @abstractmethod
    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> Union[List[str], List[dict]]:
        """
        Generates a final output or answer based on the query and retrieved context.

        Args:
            query (str): The original user query.
            query_batch (List[str]): The batch of original user queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.
            context (Optional[Any]): The formatted context string/data used to
                augment the generation. Output from get_context_from_objects.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """
        pass

    def extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        """
        Extract node_ids and edge_ids from retrieved_objects for session QA.
        Override in retrievers that use session and have graph elements to store.
        Only called when session is enabled.
        """
        return None

    def merge_retrieved_objects(self, primary: Any, secondary: Any) -> Any:
        """Combine two retrievals of this retriever's own result shape.

        Called when one turn retrieves twice — a session turn runs the raw question and a
        conversational rewrite of it — so the retriever formats context from both at once.
        Only the retriever knows its object shape, so only it can merge them.

        The default keeps whichever retrieval it has: either lane may be None when the
        other one failed, and dropping the surviving lane would discard a good result.
        """
        return primary if primary is not None else secondary

    async def append_references(self, completions: List[Any], retrieved_objects: Any) -> List[Any]:
        """Apply retriever-owned references; unsupported retrievers leave answers unchanged."""
        return completions

    def get_context_evidence(
        self,
        retrieved_objects: Any,
        dataset_id: Any = None,
    ) -> List["EvidenceReference"]:
        """Return structured identifiers for artifacts included in completion context.

        Retrievers opt in by overriding this pure, synchronous hook. The default
        intentionally returns no evidence so community retrievers remain compatible.
        """
        return []

    async def prepare_session_turn_for_retrieval(self, query: str):
        """Analyze a session turn before retrieval and fail open to the original query."""
        try:
            from cognee.infrastructure.session.get_session_manager import get_session_manager
            from cognee.infrastructure.session.session_manager import SessionTurnPreparation

            if not query:
                return SessionTurnPreparation(should_answer=True, effective_query=query or "")

            session_manager = get_session_manager()
            return await session_manager.prepare_session_turn(
                session_id=getattr(self, "session_id", None),
                query=query,
            )
        except Exception:
            from cognee.infrastructure.session.session_manager import SessionTurnPreparation

            return SessionTurnPreparation(should_answer=True, effective_query=query or "")

    async def get_completion(self, query: str) -> Union[List[str], List[dict]]:
        """
        Generates a final output or answer based on the query and retrieved context.

        Args:
            query (str): The original user query.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """
        retrieved_objects = await self.get_retrieved_objects(query=query)
        context = await self.get_context_from_objects(
            query=query, retrieved_objects=retrieved_objects
        )
        completion = await self.get_completion_from_context(
            query=query, retrieved_objects=retrieved_objects, context=context
        )
        return completion
