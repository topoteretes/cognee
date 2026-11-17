from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from cognee.modules.retrieval.utils.resolved_context import user_resolution_scope
from cognee.shared.logging_utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from cognee.modules.search.models.EvidenceReference import EvidenceReference


class BaseRetriever(ABC):
    """
    Extension point for search: each ``SearchType`` is served by one retriever.

    ``search()`` and ``recall()`` reach a retriever through the shared search flow, which
    has already resolved the datasets the caller may read and, with backend access control
    on (the default), bound each searched dataset's graph and vector databases, so
    ``get_graph_engine()`` and the vector engine inside a retriever read that dataset. The
    flow then drives the three steps below and builds the result: session-turn analysis
    that may rewrite the query, source-document access timestamps (with
    ``ENABLE_LAST_ACCESSED=true``), the ``only_context`` prompt preview, structured
    evidence for ``include_references=True``, and tracing spans.

    A subclass supplies the three async steps:

    1. ``get_retrieved_objects``: fetch raw results (graph edges, vector hits, chunks).
    2. ``get_context_from_objects``: format them as the context an LLM will read.
    3. ``get_completion_from_context``: answer from that context, or return the context
       unchanged for a non-generative retriever.

    A custom class is registered with ``use_retriever(search_type, cls)``
    (``register_retriever.py``) and takes precedence over the built-in retriever for that
    type. Four cases are decided before registered classes are looked up:

    - ``AGENTIC_COMPLETION`` always uses the built-in retriever.
    - ``CYPHER`` and ``NATURAL_LANGUAGE`` raise ``SearchTypeDisabledError`` unless
      ``ALLOW_CYPHER_QUERY=true``.
    - A ``HYBRID_COMPLETION`` request the hybrid lanes cannot serve (a custom
      ``node_type``, ``neighborhood_depth``, ``feedback_influence`` above 0, or no chunk
      collection) runs as ``GRAPH_COMPLETION``; ``HYBRID_COMPLETION_DECOMPOSITION`` raises
      ``QueryValidationError`` instead.
    - ``FEELING_LUCKY`` is resolved to another type first.

    The class is constructed with the search call's keyword arguments (``user``,
    ``dataset``, ``top_k``, ``node_name``, ``session_id``, ...), so its ``__init__`` should
    accept ``**kwargs``. The concurrent session turn runs only for the built-in
    ``CompletionRetriever``, ``GraphCompletionRetriever`` and ``HybridRetriever`` classes;
    every other retriever takes the sequential path, where its own
    ``get_completion_from_context`` is responsible for honouring
    ``skip_completion_on_empty_context`` and calling ``append_references``.

    The optional hooks default to no-ops, so implementing the three steps is enough; override
    them to add session graph tracking, citations and structured evidence.
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

    # When True, an empty retrieval context returns an empty result instead of
    # calling the LLM: a model asked to answer from nothing can only produce a
    # phantom "no context provided" deflection, which callers cannot tell apart
    # from a real answer (SDK-270 / gh #3728). Retrievers whose completion can
    # legitimately answer without retrieval context (e.g. tool-driven agentic
    # flows) must leave this False.
    skip_completion_on_empty_context = False

    @abstractmethod
    async def get_retrieved_objects(self, query: str | None, query_batch: str | None) -> Any:
        """
        Fetch the raw results for the query from the dataset's graph or vector store.

        The search flow passes only ``query`` (the session-rewritten query when turn
        analysis produced one), so give ``query_batch`` a default of ``None``. With
        ``ENABLE_LAST_ACCESSED=true``, the documents behind the chunks in a truthy result
        (a list of graph edges, or a dict with ``chunks`` / ``entities``) get their
        ``last_accessed`` timestamp updated.

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.

        Returns:
            Any: The raw objects (e.g., Edge objects, Document chunks) relevant to the
                query, in a shape this retriever's other methods understand.
        """

    @abstractmethod
    async def get_context_from_objects(
        self,
        query: str | None = None,
        query_batch: str | None = None,
        retrieved_objects: Any = None,
    ) -> str | list[str]:
        """
        Transforms raw retrieved objects into a structured context for the LLM.

        The result is returned as the search result's context, and with
        ``only_context=True`` a retriever that sends one templated prompt (set prompt
        paths and ``supports_prompt_preview``) has it rendered into that prompt instead
        of answered. Return an empty value when nothing was found:
        ``only_context`` then returns the empty context rather than a prompt, and
        ``skip_completion_on_empty_context`` relies on it.

        Args:
            query (str): The search query or input string.
            query_batch (List[str]): The batch of search queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.

        Returns:
            Any: The formatted context (typically a string or a list of strings)
                 to be injected into a prompt.
        """

    @abstractmethod
    async def get_completion_from_context(
        self,
        query: str | None = None,
        query_batch: list[str] | None = None,
        retrieved_objects: Any = None,
        context: Any = None,
    ) -> list[str] | list[dict]:
        """
        Generates a final output or answer based on the query and retrieved context.

        The search flow skips this step for ``only_context=True``. It passes the original
        query; declaring ``effective_query`` or ``turn_preparation`` parameters also
        receives the session-rewritten query and the turn analysis. An implementation
        that sets ``skip_completion_on_empty_context`` should return ``[]`` for an empty
        context, and one that supports citations should pass its completions through
        ``append_references``.

        Args:
            query (str): The original user query.
            query_batch (List[str]): The batch of original user queries.
            retrieved_objects (List[Any]): The output from get_retrieved_objects.
            context (Optional[Any]): The formatted context string/data used to
                augment the generation. Output from get_context_from_objects.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """

    def extract_context_object_ids(self, retrieved_objects: Any) -> dict[str, list[str]] | None:
        """
        Extract node_ids and edge_ids from retrieved_objects for session QA.

        The ids are stored with the session's QA turn, where ``improve()``'s feedback
        weighting applies that turn's feedback to the graph elements that fed the answer.
        Override in retrievers that use session and have graph elements to store. Only
        called when session is enabled: by the concurrent session turn, or by a
        retriever's own completion that passes the ids to the session manager.
        """
        return None

    def merge_retrieved_objects(self, primary: Any, secondary: Any) -> Any:
        """Combine two retrievals of this retriever's own result shape.

        Called when one turn retrieves twice — a session turn runs the raw question and a
        conversational rewrite of it — so the retriever formats context from both at once.
        Only the retriever knows its object shape, so only it can merge them. Only the
        concurrent session turn calls it, so the built-in classes listed on the class use it.

        The default keeps whichever retrieval it has: either lane may be None when the
        other one failed, and dropping the surviving lane would discard a good result.
        """
        return primary if primary is not None else secondary

    async def append_references(self, completions: list[Any], retrieved_objects: Any) -> list[Any]:
        """Apply retriever-owned references; unsupported retrievers leave answers unchanged.

        Called on the answers of a concurrent session turn, and by built-in completions on
        their own answers, so citations appear whichever path produced the answer.
        """
        return completions

    def get_context_evidence(
        self,
        retrieved_objects: Any,
        dataset_id: Any = None,
    ) -> list["EvidenceReference"]:
        """Return structured identifiers for artifacts included in completion context.

        Retrievers opt in by overriding this pure, synchronous hook. The default
        intentionally returns no evidence so community retrievers remain compatible.

        With ``include_references=True`` the search flow calls it with the searched
        dataset's id and returns the references as the result's ``evidence``; when a
        dataset is set, graph-edge references are then extended with their recorded source
        evidence. An exception here is logged and treated as no evidence, so it never
        fails the search.
        """
        return []

    async def prepare_session_turn_for_retrieval(self, query: str):
        """Analyze a session turn before retrieval and fail open to the original query.

        The sequential search path calls it unless ``only_context`` is set or the class
        sets ``supports_session_turn_preparation = False``. A turn judged not to need an
        answer is acknowledged without retrieval; otherwise its ``effective_query`` drives
        retrieval and context.
        """
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
            logger.debug(
                "Falling back after error in BaseRetriever.prepare_session_turn_for_retrieval",
                exc_info=True,
            )
            from cognee.infrastructure.session.session_manager import SessionTurnPreparation

            return SessionTurnPreparation(should_answer=True, effective_query=query or "")

    @user_resolution_scope()
    async def get_completion(self, query: str) -> list[str] | list[dict]:
        """
        Run the three steps directly for one query, outside the search flow.

        Useful for calling a retriever on its own. Unlike ``search()``, it checks no
        permissions, binds no dataset databases (it reads whichever are current), builds
        no ``include_references`` evidence, and retrieves with the raw query: no turn
        analysis runs before retrieval.

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
