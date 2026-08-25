import asyncio
import re
from typing import Any, Callable, Optional, List, Union
from heapq import nlargest

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.shared.logging_utils import get_logger


logger = get_logger("LexicalRetriever")


def nodeset_tags(payload: dict) -> list[str]:
    """Return a node payload's ``belongs_to_set`` tags as NodeSet names.

    Graph serialization already reduces the field to names, and keeps it as a node
    property rather than only as edges precisely so node sets can be filtered on
    (see ``get_graph_from_model``). A tag can still reach the graph as a mapping or
    a DataPoint from another writer, so each shape is reduced to the same string
    key the vector payloads carry.
    """
    value = payload.get("belongs_to_set")
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    tags: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            tag: Any = item
        elif isinstance(item, dict):
            tag = item.get("name") or item.get("id")
        else:
            tag = getattr(item, "name", None) or getattr(item, "id", None)
        if tag is not None:
            tags.append(str(tag))
    return tags


def matches_node_set(
    payload: dict,
    node_name: Optional[List[str]],
    node_name_filter_operator: str = "OR",
) -> bool:
    """Whether a chunk payload belongs to the requested node sets.

    Mirrors what the vector adapters apply to the same field: "OR" keeps a chunk
    tagged with any of the requested sets, "AND" only one tagged with all of them.
    An empty ``node_name`` applies no filter, exactly as it does on the vector path.
    """
    if not node_name:
        return True

    tags = set(nodeset_tags(payload))
    wanted = {str(name) for name in node_name}

    if node_name_filter_operator == "AND":
        return wanted.issubset(tags)
    return bool(tags & wanted)


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
        self,
        tokenizer: Callable,
        scorer: Callable,
        top_k: int = 15,
        with_scores: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ):
        if not callable(tokenizer) or not callable(scorer):
            raise TypeError("tokenizer and scorer must be callables")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if node_name is not None and not isinstance(node_name, (list, tuple)):
            raise TypeError("node_name must be a list of node set names")

        self.tokenizer = tokenizer
        self.scorer = scorer
        self.top_k = top_k
        self.with_scores = bool(with_scores)
        self.node_name = list(node_name) if node_name else None
        self.node_name_filter_operator = node_name_filter_operator

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
            document_chunk_count = 0
            for node in nodes:
                try:
                    chunk_id, document = node
                except Exception:
                    logger.warning("Skipping node with unexpected shape: %r", node)
                    continue

                if document.get("type") == "DocumentChunk" and document.get("text"):
                    document_chunk_count += 1
                    # Filter before tokenizing so a scoped search neither pays to
                    # tokenize chunks it will not rank, nor lets them influence
                    # corpus statistics a subclass derives from what was loaded.
                    if not matches_node_set(
                        document, self.node_name, self.node_name_filter_operator
                    ):
                        continue
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
                if self.node_name and document_chunk_count:
                    # There is data, the requested node sets are simply empty. The
                    # vector chunk search returns nothing in that case rather than
                    # reporting that the system holds no data, so this does too.
                    logger.info(
                        "No chunks matched node sets %s out of %d document chunks",
                        self.node_name,
                        document_chunk_count,
                    )
                    self._initialized = True
                    return
                logger.error("Initialization completed but no valid chunks were loaded.")
                raise NoDataError("No valid chunks loaded during initialization.")

            self._initialized = True
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
