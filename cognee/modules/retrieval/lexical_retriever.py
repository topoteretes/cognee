from typing import Any, Optional, Callable
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.shared.logging_utils import get_logger
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.storage.utils import get_own_properties
import asyncio
import json
from cognee.modules.retrieval.exceptions.exceptions import NoDataError

logger = get_logger("LexicalRetriever")


class LexicalRetriever(BaseRetriever):
    """
    Retriever for retrieving semantic search results from pre-tokenized DocumentChunks.

    Public methods:
        - get_context: Retrieves relevant context using a similarity scorer.
        - get_completion: Returns the retrieved context.
    """

    # Shared across all retrievers
    _chunks: dict[str, Any] = {}
    _payloads: dict[str, Any] = {}
    _initialized: bool = False
    _init_lock = asyncio.Lock()

    def __init__(self, tokenizer: Callable, scorer: Callable, top_k: int = 10, with_scores: bool = False):
        """
        Parameters:
        -----------
        - tokenizer (Callable): Function that takes text and returns a list of tokens.
        - scorer (Callable): Function that takes (query_tokens, chunk_tokens) and returns a similarity score.
        - top_k (int): Number of top chunks to retrieve.
        - with_scores (bool): Whether to return (chunk, score) pairs instead of just chunks.
        """
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.top_k = top_k
        self.with_scores = with_scores

    def fix_json_strings(self, obj):
        """
        Recursively convert any JSON string values into dict/list.
        """
        if isinstance(obj, dict):
            return {k: self.fix_json_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.fix_json_strings(item) for item in obj]
        elif isinstance(obj, str):
            try:
                return self.fix_json_strings(json.loads(obj))
            except (json.JSONDecodeError, TypeError):
                return obj
        return obj

    @classmethod
    async def initialize(cls, tokenizer: Callable):
        """
        Initialize retriever by reading all DocumentChunks from graph_engine.
        Tokenizes text and stores them in shared class-level cache.
        """
        if cls._initialized:
            return

        async with cls._init_lock:
            if cls._initialized:
                return

            try:
                graph_engine = await get_graph_engine()
                nodes, _ = await graph_engine.get_graph_data()
            except Exception as e:
                logger.error("Graph engine initialization failed: %s", str(e))
                raise NoDataError("Graph engine initialization failed")

            chunk_count = 0
            for node in nodes:
                chunk_id, document = node
                fixed_document = cls.fix_json_strings(cls, document)
                if fixed_document.get("type") == "DocumentChunk" and fixed_document.get("text"):
                    try:
                        tokens = tokenizer(fixed_document["text"])
                        document_chunk = DocumentChunk.from_dict(fixed_document)
                        new_document = get_own_properties(document_chunk)
                        new_document["id"] = str(new_document.get("id"))
                        cls._chunks[str(chunk_id)] = tokens
                        cls._payloads[str(chunk_id)] = new_document
                        chunk_count += 1
                    except Exception as e:
                        logger.error("Tokenizer failed for chunk %s: %s", chunk_id, str(e))

            if chunk_count == 0:
                logger.error("Initialization completed but no valid chunks were loaded.")
                raise NoDataError("No valid chunks loaded during initialization.")

            cls._initialized = True
            logger.info("Initialized with %d document chunks", len(cls._chunks))

    async def get_context(self, query: str) -> Any:
        """
        Retrieves relevant chunks for the given query.
        """
        if not self.__class__._initialized:
            await self.__class__.initialize(self.tokenizer)

        if not self.__class__._chunks:
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
        for chunk_id, chunk_tokens in self.__class__._chunks.items():
            try:
                score = self.scorer(query_tokens, chunk_tokens)
                if not isinstance(score, (int, float)):
                    logger.warning("Non-numeric score for chunk %s → treated as 0.0", chunk_id)
                    score = 0.0
            except Exception as e:
                logger.error("Scorer failed for chunk %s: %s", chunk_id, str(e))
                score = 0.0
            results.append((chunk_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[: self.top_k]

        logger.info(
            "Retrieved %d/%d chunks for query (len=%d)",
            len(top_results), len(results), len(query_tokens)
        )

        if self.with_scores:
            return [(self.__class__._payloads[chunk_id], score) for chunk_id, score in top_results]
        else:
            return [self.__class__._payloads[chunk_id] for chunk_id, _ in top_results]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Returns context for the given query (retrieves if not provided).
        """
        if context is None:
            context = await self.get_context(query)
        return context
