import asyncio
from typing import Any, Callable, Optional
from heapq import nlargest

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger


logger = get_logger("LexicalRetriever")


class LexicalRetriever(BaseRetriever):
    def __init__(
        self, tokenizer: Callable, scorer: Callable, top_k: int = 10, with_scores: bool = False
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
        """Initialize retriever by reading all DocumentChunks from graph_engine."""
        async with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing LexicalRetriever by loading DocumentChunks from graph engine")

            try:
                graph_engine = await get_graph_engine()
                nodes, _ = await graph_engine.get_filtered_graph_data([{"type": ["DocumentChunk"]}])
            except Exception as e:
                logger.error("Graph engine initialization failed")
                raise NoDataError("Graph engine initialization failed") from e

            chunk_count = 0
            for node in nodes:
                try:
                    chunk_id, document = node
                except Exception:
                    logger.warning("Skipping node with unexpected shape: %r", node)
                    continue

                if document.get("type") == "DocumentChunk" and document.get("text"):
                    try:
                        tokens = self.tokenizer(document["text"])
                        if not tokens:
                            continue
                        self.chunks[str(document.get("id", chunk_id))] = tokens
                        self.payloads[str(document.get("id", chunk_id))] = document
                        chunk_count += 1
                    except Exception as e:
                        logger.error("Tokenizer failed for chunk %s: %s", chunk_id, str(e))

            if chunk_count == 0:
                logger.error("Initialization completed but no valid chunks were loaded.")
                raise NoDataError("No valid chunks loaded during initialization.")

            self._initialized = True
            logger.info("Initialized with %d document chunks", len(self.chunks))

    async def get_context(self, query: str) -> Any:
        """Retrieves relevant chunks for the given query."""
        if not self._initialized:
            await self.initialize()

        if not self.chunks:
            logger.warning("No chunks available in retriever")
            return []

        try:
            query_tokens = self.tokenizer(query)
        except Exception as e:
            logger.error("Failed to tokenize query: %s", str(e))
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
            except Exception as e:
                logger.error("Scorer failed for chunk %s: %s", chunk_id, str(e))
                score = 0.0
            results.append((chunk_id, score))

        top_results = nlargest(self.top_k, results, key=lambda x: x[1])
        logger.info(
            "Retrieved %d/%d chunks for query (len=%d)",
            len(top_results),
            len(results),
            len(query_tokens),
        )

        if self.with_scores:
            return [(self.payloads[chunk_id], score) for chunk_id, score in top_results]
        else:
            return [self.payloads[chunk_id] for chunk_id, _ in top_results]

    async def get_completion(
        self, query: str, context: Optional[Any] = None, session_id: Optional[str] = None
    ) -> Any:
        """
        Returns context for the given query (retrieves if not provided).

        Parameters:
        -----------

            - query (str): The query string to retrieve context for.
            - context (Optional[Any]): Optional pre-fetched context; if None, it retrieves
              the context for the query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The context, either provided or retrieved.
        """
        if context is None:
            context = await self.get_context(query)
        return context
