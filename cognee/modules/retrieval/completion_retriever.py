from typing import Any, Dict, List, Optional, Type

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.merge_results import conversational_reserve, merge_ranked
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.used_graph_elements import extract_from_scored_results
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.modules.retrieval.utils.references import append_chunk_evidence
from cognee.base_config import get_base_config
from cognee.modules.user_preferences import (
    load_preference_text,
    load_preference_weights,
    personal_factor,
)
from cognee.modules.retrieval.utils.evidence import chunk_context_evidence

logger = get_logger("CompletionRetriever")


def _stable_sort_by_personal_distance(
    found_chunks: List[Any], weights: Dict[str, float], influence: float
) -> List[Any]:
    """Stable re-sort of ScoredResult chunks by personalized distance.

    ``score`` is a distance here (lower is better), so a preferred chunk's
    distance shrinks by ``personal_factor(..., distance_space=True)``. Chunk
    id comes from ``payload["id"]`` falling back to ``.id`` — the same rule
    ``extract_from_scored_results`` uses, so the ids match the ones the
    preference update wrote ``prefers`` edges for. Chunks with no matching
    weight keep their raw distance, and the sort is stable, so ties keep the
    vector engine's order.
    """

    def personalized_distance(chunk: Any) -> float:
        chunk_id = None
        payload = getattr(chunk, "payload", None)
        if isinstance(payload, dict):
            chunk_id = payload.get("id")
        if chunk_id is None:
            chunk_id = getattr(chunk, "id", None)
        weight = weights.get(str(chunk_id)) if chunk_id is not None else None
        if weight is None:
            return chunk.score
        return chunk.score * personal_factor(weight, influence, distance_space=True)

    return sorted(found_chunks, key=personalized_distance)


async def _weights_matching_collection(
    vector_engine: Any, weights: Dict[str, float], collection_name: str = "DocumentChunk_text"
) -> Dict[str, float]:
    """Keep only the prefers weights whose key is a row in this collection.

    Weight keys span every rated node — graph entities included — but only
    chunk ids can ever match a ``DocumentChunk_text`` result, so weights that
    point elsewhere must not trigger the wide fetch: the extra work would be
    guaranteed to change nothing. One id-lookup ``retrieve`` answers the
    question; on any error it fails open by returning the weights unfiltered,
    so a broken lookup costs a wider search, never a lost personalization.
    """
    try:
        rows = await vector_engine.retrieve(collection_name, list(weights))
    except Exception as error:
        logger.debug("Preference weight collection check failed open: %s", error)
        return weights

    present = set()
    for row in rows:
        payload = getattr(row, "payload", None)
        if isinstance(payload, dict) and payload.get("id") is not None:
            present.add(str(payload["id"]))
        row_id = getattr(row, "id", None)
        if row_id is not None:
            present.add(str(row_id))
    return {key: weight for key, weight in weights.items() if key in present}


class CompletionRetriever(BaseRetriever):
    """
    Retriever for handling LLM-based completion searches.
    """

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 1,
        session_id: Optional[str] = None,
        response_model: Type = str,
        include_references: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 1
        self.wide_search_top_k = wide_search_top_k
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.response_model = response_model
        self.include_references = include_references
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator

    async def get_retrieved_objects(self, query: str) -> Any:
        vector_engine = await get_vector_engine_async()

        # Loaded before the search because it decides the limit: with prefers
        # weights that can match this collection we widen the fetch to
        # wide_search_top_k — the one knob the search path already owns for
        # "how many candidates before trimming" — so personalization can change
        # which chunks make the cut, not just their order. Flag off, no node,
        # or weights that only point at graph entities keep limit=top_k and no
        # re-sort — byte-identical to the un-personalized path.
        weights = await load_preference_weights()
        if weights:
            weights = await _weights_matching_collection(vector_engine, weights)
        limit = max(self.top_k, self.wide_search_top_k or 0) if weights else self.top_k

        try:
            found_chunks = await vector_engine.search(
                "DocumentChunk_text",
                query,
                limit=limit,
                include_payload=True,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
            )
        except CollectionNotFoundError as error:
            logger.error("DocumentChunk_text collection not found")
            raise NoDataError("No data found in the system, please add data first.") from error

        if weights:
            # Re-sort before merge_retrieved_objects: the session path merges
            # two retrieval lanes with merge_ranked(..., limit=top_k), so
            # ordering must already be personalized when the merge trims.
            influence = get_base_config().personalization_influence
            found_chunks = _stable_sort_by_personal_distance(found_chunks, weights, influence)[
                : self.top_k
            ]

        return found_chunks

    def merge_retrieved_objects(self, primary: Any, secondary: Any) -> Any:
        return merge_ranked(
            primary,
            secondary,
            limit=self.top_k,
            secondary_reserve=conversational_reserve(self.top_k),
        )

    def extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        """Extract node_ids from ScoredResult-like list for session QA."""
        if isinstance(retrieved_objects, list) and retrieved_objects:
            return extract_from_scored_results(retrieved_objects)
        return None

    def get_context_evidence(self, retrieved_objects: Any, dataset_id: Any = None):
        """Return the exact chunks concatenated into this RAG completion's context."""
        return chunk_context_evidence(retrieved_objects, dataset_id=dataset_id)

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves relevant document chunks as context.

        Fetches document chunks based on a query from a vector engine and combines their text.
        Returns empty string if no chunks are found. Raises NoDataError if the collection is not
        found.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunks.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved document chunks, or an
              empty string if none are found.
        """
        if retrieved_objects:
            # Combine all chunks text returned from vector search (number of chunks is determined by top_k)
            chunks_payload = [found_chunk.payload["text"] for found_chunk in retrieved_objects]
            combined_context = "\n".join(chunks_payload)
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
        # Sessionless guidance site: preference text rides the guidance channel
        # (conversation_history), never context. The lookup is memoized per
        # context; this sessionless path runs retrieval and completion in one
        # context, so this reuses the get_retrieved_objects read. (Across a
        # task fan-out that sharing needs warm_preference_cache in the parent
        # — the ContextVar does not propagate out of gather lanes.) Empty text
        # is falsy and leaves the system prompt untouched. The session path
        # never reaches this method, so it cannot collide with the session
        # guidance block, which owns preference rendering on that path.
        preference_text = await load_preference_text()
        completion = await generate_completion(
            query=query, conversation_history=preference_text, **kwargs
        )
        return [completion]

    async def append_references(self, completions: List[Any], retrieved_objects: Any) -> List[Any]:
        return append_chunk_evidence(
            completions,
            retrieved_objects,
            enabled=self.include_references and self.response_model is str,
        )

    async def get_completion_from_context(
        self,
        query: str,
        retrieved_objects: Any,
        context: Optional[Any] = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
    ) -> List[Any]:
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
            used_graph_element_ids = self.extract_context_object_ids(retrieved_objects)
            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
                summarize_context=False,
                used_graph_element_ids=used_graph_element_ids,
                max_context_chars=getattr(self, "max_context_chars", None),
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
            completions = [completion]
        else:
            completions = await self._generate_completion_without_session(query, context)

        # Both the session/cache branch and the non-session branch rejoin here so
        # logged-in/cached calls also receive references. Evidence is grounded in
        # each completion's own text, so a cache-hit answer never cites chunks
        # that share nothing with it.
        return await self.append_references(completions, retrieved_objects)
