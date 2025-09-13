from typing import Any, Optional, Callable
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.shared.logging_utils import get_logger
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.storage.utils import get_own_properties
import asyncio
import json
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from heapq import nlargest
logger = get_logger("LexicalRetriever")

class LexicalRetriever(BaseRetriever):
    _global_chunks: dict[str, Any] = {}
    _global_payloads: dict[str, Any] = {}
    _global_initialized = False
    _global_init_lock = asyncio.Lock()

    def __init__(self, tokenizer: Callable, scorer: Callable, top_k: int = 10, with_scores: bool = False):
        if not callable(tokenizer) or not callable(scorer):
            raise TypeError("tokenizer and scorer must be callables")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.tokenizer = tokenizer
        self.scorer = scorer
        self.top_k = top_k
        self.with_scores = bool(with_scores)

    def fix_json_strings(self, obj):
        """
        Recursively convert any JSON string values into dict/list.
        """
        if isinstance(obj, dict):
            return {k: self.fix_json_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.fix_json_strings(item) for item in obj]
        elif isinstance(obj, str):
            s = obj.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    return self.fix_json_strings(json.loads(s))
                except (json.JSONDecodeError, TypeError):
                    pass
            return obj
        return obj
    async def initialize(self):
        if self.__class__._global_initialized:
            return  # already loaded

        async with self.__class__._global_init_lock:
            if self.__class__._global_initialized:
                return

            logger.info("Initializing LexicalRetriever (shared) from graph engine")
            graph_engine = await get_graph_engine()
            nodes, _ = await graph_engine.get_graph_data()

            chunk_count = 0
            for node in nodes:
                try:
                    chunk_id, document = node
                except Exception:
                    continue

                fixed_document = self.fix_json_strings(document)
                if fixed_document.get("type") == "DocumentChunk" and fixed_document.get("text"):
                    try:
                        tokens = self.tokenizer(fixed_document["text"])
                        if not tokens:
                            continue
                        document_chunk = DocumentChunk.from_dict(fixed_document)
                        new_document = get_own_properties(document_chunk)
                        new_document["id"] = str(new_document.get("id", chunk_id))
                        self.__class__._global_chunks[str(chunk_id)] = tokens
                        self.__class__._global_payloads[str(chunk_id)] = new_document
                        chunk_count += 1
                    except Exception as e:
                        logger.error("Tokenizer failed for chunk %s: %s", chunk_id, str(e))

            if chunk_count == 0:
                raise NoDataError("No valid chunks loaded during initialization.")

            self.__class__._global_initialized = True
            logger.info("Shared retriever initialized with %d chunks", chunk_count)

    async def get_context(self, query: str) -> Any:
        if not self.__class__._global_initialized:
            await self.initialize()

        if not self.__class__._global_chunks:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        results = []
        for chunk_id, chunk_tokens in self.__class__._global_chunks.items():
            try:
                score = self.scorer(query_tokens, chunk_tokens)
                if not isinstance(score, (int, float)):
                    score = 0.0
            except Exception:
                score = 0.0
            results.append((chunk_id, score))

        top_results = nlargest(self.top_k, results, key=lambda x: x[1])

        if self.with_scores:
            return [(self.__class__._global_payloads[chunk_id], score) for chunk_id, score in top_results]
        else:
            return [self.__class__._global_payloads[chunk_id] for chunk_id, _ in top_results]

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Returns context for the given query (retrieves if not provided).
        """
        if context is None:
            context = await self.get_context(query)
        return context
