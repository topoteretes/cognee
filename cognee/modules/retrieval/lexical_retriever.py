import asyncio
import re
from typing import Any, Callable, Optional, List, Union
from heapq import nlargest

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.retrieval.utils import lexical_corpus_cache
from cognee.shared.logging_utils import get_logger


logger = get_logger("LexicalRetriever")


def tokenize_words(text: str, stop_words: Optional[set[str]] = None) -> list[str]:
    """Lowercase, split on word characters, and drop any stop words.

    Shared by the lexical retrievers so tokenization stays consistent across scorers.
    """
    tokens = re.findall(r"\w+", text.lower())
    if not stop_words:
        return tokens
    return [token for token in tokens if token not in stop_words]


class LexicalRetriever(BaseRetriever):
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
        """Initialize retriever from the corpus cache, loading from the graph on a miss."""
        async with self._init_lock:
            if self._initialized:
                return

            cache_key = self._corpus_cache_key()
            if cache_key is None:
                await self._load_corpus()
                self._initialized = True
                return

            if self._restore_if_cached(cache_key):
                return

            async with lexical_corpus_cache.lock(cache_key):
                if self._restore_if_cached(cache_key):
                    return
                await self._load_corpus()
                lexical_corpus_cache.put(cache_key, self._cache_state())
                self._initialized = True

    def _restore_if_cached(self, cache_key) -> bool:
        state = lexical_corpus_cache.get(cache_key)
        if state is None:
            return False
        self._restore_cache_state(state)
        self._initialized = True
        return True

    def _corpus_cache_key(self) -> Optional[tuple]:
        """Cache key scoped to the current graph context, or None when not cacheable."""
        tokenizer_key = self._tokenizer_cache_key()
        if tokenizer_key is None:
            return None
        graph_config = get_graph_context_config()
        config_key = tuple(sorted((str(key), str(value)) for key, value in graph_config.items()))
        return (config_key, type(self).__name__, tokenizer_key)

    def _tokenizer_cache_key(self) -> Optional[tuple]:
        """Subclasses with a hashable tokenizer config opt into corpus caching here.

        The base class accepts arbitrary tokenizer callables, which cannot be keyed
        safely, so it returns None and skips the cache.
        """
        return None

    def _cache_state(self) -> dict:
        return {"chunks": self.chunks, "payloads": self.payloads}

    def _restore_cache_state(self, state: dict) -> None:
        self.chunks = state["chunks"]
        self.payloads = state["payloads"]

    async def _load_corpus(self):
        """Read all DocumentChunks from the graph engine and tokenize them."""
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
                    # Some graph adapters (e.g. kuzu) omit "id" from node payloads;
                    # downstream consumers match chunks across channels by payload id.
                    document_id = str(document.get("id") or chunk_id)
                    document.setdefault("id", document_id)
                    self.chunks[document_id] = tokens
                    self.payloads[document_id] = document
                    chunk_count += 1
                except Exception as e:
                    logger.error("Tokenizer failed for chunk %s: %s", chunk_id, str(e))

        if chunk_count == 0:
            logger.error("Initialization completed but no valid chunks were loaded.")
            raise NoDataError("No valid chunks loaded during initialization.")

        logger.info("Initialized with %d document chunks", len(self.chunks))

    async def get_retrieved_objects(self, query: str) -> Any:
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
        if retrieved_objects:
            payload_texts = [payload["text"] for payload in retrieved_objects]
            return "\n".join(payload_texts)
        else:
            return ""

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> Union[List[str], List[dict]]:
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
        return retrieved_objects
