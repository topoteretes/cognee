import asyncio
from collections.abc import Callable
from heapq import nlargest
from typing import Any

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions.exceptions import (
    LexicalSearchNotSupportedError,
)
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.utils.conflict_context import (
    attach_chunk_conflicts,
    public_chunk_payload,
    render_chunk_context,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("LexicalRetriever")


# The tokenizer moved next to the backend lexical indexes so every engine
# produces the same tokens; the name stays importable from here.
from cognee.infrastructure.databases.vector.lexical import tokenize_words  # noqa: E402

__all__ = ["LexicalRetriever", "tokenize_words"]


class LexicalRetriever(BaseRetriever):
    # Lexical search scores chunks locally and never calls an LLM, so the conversational
    # session-turn analysis would add a pre-retrieval LLM round trip to an otherwise
    # sub-second, deterministic path. Opt out, like the other non-generative retrievers.
    # Inherited by BM25ChunksRetriever and JaccardChunksRetriever.
    supports_session_turn_preparation = False

    # The graph node type scored in memory, and the vector-store collection
    # whose lexical (BM25) index answers instead on backends that keep one
    # (``VectorDBInterface.supports_lexical_search``). None, the default,
    # keeps scoring in memory: a custom scorer has no backend equivalent, so
    # only the BM25 subclasses name a collection.
    node_type = "DocumentChunk"
    collection_name: str | None = None
    # Whether hits are chunks whose conflicts (contradiction edges) are attached.
    attaches_conflicts = True

    def __init__(
        self, tokenizer: Callable, scorer: Callable, top_k: int = 15, with_scores: bool = False
    ):
        if not callable(tokenizer) or not callable(scorer):
            raise TypeError("tokenizer and scorer must be callables")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.tokenizer = tokenizer
        self.scorer = scorer
        self.top_k = top_k
        self.with_scores = bool(with_scores)

        # Cache keyed by dataset context
        self.chunks: dict[str, Any] = {}  # {chunk_id: tokens}
        self.payloads: dict[str, Any] = {}  # {chunk_id: original_document}
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize retriever by reading all nodes of ``node_type`` from graph_engine."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info(
                "Initializing LexicalRetriever by loading %s nodes from graph engine",
                self.node_type,
            )

            try:
                graph_engine = await get_graph_engine()
                nodes, _ = await graph_engine.get_filtered_graph_data([{"type": [self.node_type]}])
            except Exception as e:
                logger.error("Graph engine initialization failed")
                raise NoDataError("Graph engine initialization failed") from e

            chunk_count = 0
            for node in nodes:
                try:
                    chunk_id, document = node
                except Exception:
                    logger.warning("Skipping node with unexpected shape: %r", node, exc_info=True)
                    continue

                if document.get("type") == self.node_type and document.get("text"):
                    try:
                        tokens = self.tokenizer(document["text"])
                        if not tokens:
                            continue
                        # Some graph adapters (e.g. kuzu) omit "id" from node payloads;
                        # downstream consumers match chunks across channels by payload id.
                        document_id = str(document.get("id") or chunk_id)
                        document.setdefault("id", document_id)
                        self.chunks[document_id] = tokens
                        self.payloads[document_id] = document
                        chunk_count += 1
                    except Exception:
                        logger.exception("Tokenizer failed for chunk %s", chunk_id)

            if chunk_count == 0:
                logger.error("Initialization completed but no valid chunks were loaded.")
                raise NoDataError("No valid chunks loaded during initialization.")

            self._initialized = True
            logger.info("Initialized with %d document chunks", len(self.chunks))

    async def _native_hits(self, query: str) -> list[tuple[dict, float]] | None:
        """BM25 hits from the vector store's lexical index, or None when there is none.

        The backend tokenizes the query itself (the same way it tokenized the
        rows), scores every row of the collection and returns the best
        ``top_k``; nothing is loaded into this process.
        """
        if not self.collection_name:
            return None
        try:
            vector_engine = await get_vector_engine_async()
        except Exception:
            logger.warning("Vector engine unavailable; scoring in memory", exc_info=True)
            return None
        if getattr(vector_engine, "supports_lexical_search", False) is not True:
            return None
        try:
            results = await vector_engine.lexical_search(
                self.collection_name, query, limit=self.top_k
            )
        except LexicalSearchNotSupportedError:
            return None
        hits: list[tuple[dict, float]] = []
        for result in results:
            payload = dict(result.payload or {})
            payload.setdefault("id", str(result.id))
            hits.append((payload, float(result.score)))
        logger.info("Retrieved %d hits from the %s lexical index", len(hits), self.collection_name)
        return hits

    async def _memory_hits(self, query: str) -> list[tuple[dict, float]]:
        """Score every loaded node in this process; the path without a backend index."""
        if not self._initialized:
            await self.initialize()

        if not self.chunks:
            logger.warning("No chunks available in retriever")
            return []

        try:
            query_tokens = self.tokenizer(query)
        except Exception:
            logger.exception("Failed to tokenize query")
            return []

        if not query_tokens:
            logger.warning("Query produced no tokens")
            return []

        results = []
        for chunk_id, chunk_tokens in self.chunks.items():
            try:
                score = self.scorer(query_tokens, chunk_tokens)
                if not isinstance(score, (int, float)):
                    logger.warning("Non-numeric score for chunk %s → treated as 0.0", chunk_id)
                    score = 0.0
            except Exception:
                logger.exception("Scorer failed for chunk %s", chunk_id)
                score = 0.0
            results.append((chunk_id, score))

        top_results = nlargest(self.top_k, results, key=lambda x: x[1])
        logger.info(
            "Retrieved %d/%d chunks for query (len=%d)",
            len(top_results),
            len(results),
            len(query_tokens),
        )

        # Copy: these payload dicts are this retriever's cache, shared across
        # queries, and conflict annotation writes into the hits it is handed.
        return [(dict(self.payloads[chunk_id]), score) for chunk_id, score in top_results]

    async def _collect_hits(self, query: str) -> list[tuple[dict, float]]:
        """The scored hits for ``query``: the backend index when there is one, else memory."""
        native = await self._native_hits(query)
        if native is not None:
            return native
        return await self._memory_hits(query)

    async def get_retrieved_objects(self, query: str) -> Any:
        """Retrieves relevant chunks for the given query."""
        scored_hits = await self._collect_hits(query)
        if self.attaches_conflicts and scored_hits:
            await attach_chunk_conflicts(scored_hits)
        if self.with_scores:
            return scored_hits
        return [hit_payload for hit_payload, _ in scored_hits]

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> str:
        """
        Retrieves context from retrieved chunks, in text form.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant document chunk payloads.
            - retrieved_objects (Any): The retrieved objects to be used for generating textual context.

        Returns:
        --------

            - str: A string containing the combined text of the retrieved chunk payloads, or an
              empty string if none are found.
        """
        # render_chunk_context reads the payload out of either hit shape.
        return render_chunk_context(retrieved_objects) if retrieved_objects else ""

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> list[str] | list[dict]:
        """
        Returns a completion for the given query.

        In case of the Lexical Retriever, we do not generate a completion, we just return
        the scored chunk payloads, i.e. the retrieved objects.

        Parameters:
        -----------

            - query (str): The query string to retrieve context for.
            - context (Optional[Any]): Optional pre-fetched context; if None, it retrieves
              the context for the query. (default None)

        Returns:
        --------

            - List[dict]: The retrieved objects, i.e. the scored payloads.
        """
        # TODO: Do we want to generate a completion using LLM here?
        if not retrieved_objects:
            return retrieved_objects
        if self.with_scores:
            return [(public_chunk_payload(payload), score) for payload, score in retrieved_objects]
        return [public_chunk_payload(payload) for payload in retrieved_objects]
