"""BM25 over the cached documents: the search that answers first (SDK-397).

``DOCUMENTS_LEXICAL`` is served by the caching pipeline's output: one
``CachedDocument`` per document, written before anything is chunked or
embedded. The ranking is the same Okapi BM25 the chunk retriever uses, either
in the vector store's lexical index (``CachedDocument_text`` on backends with
``supports_lexical_search``) or in memory over the ``CachedDocument`` graph
nodes. A hit is one document; its ``text`` is the passage with the most query
terms, never the whole document, so a result stays readable and bounded.
"""

import unicodedata
from typing import Any

from cognee.infrastructure.databases.vector.lexical import split_passages, tokenize_words
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.hybrid.results import payload_matches_node_filter
from cognee.modules.retrieval.utils.conflict_context import public_chunk_payload

# The snippet window: about a paragraph and a half.
SNIPPET_TOKENS = 220

# Keys a document hit carries, whatever the backend: the vector stores keep
# different payloads (LanceDB the index schema, pgvector the whole point, the
# in-memory path the graph node), so a hit is built from what they all have.
_DOCUMENT_KEYS = ("id", "document_id", "belongs_to_set")

# With a node-set filter the backend index ranks the whole collection and the
# filter runs on what it returns, so it is asked for more rows than top_k.
_FILTERED_FETCH_FACTOR = 10
_FILTERED_FETCH_MAX = 1000


# Suffixes cut by _loose, longest first; "ies"/"ied" become "y".
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s")


def _fold(token: str) -> str:
    """``token`` without accents, as LanceDB's ascii folding indexes it."""
    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _loose(token: str) -> str:
    """``token`` reduced for snippet matching: accents folded, one common English suffix cut.

    Not a stemmer: just enough that a passage the index matched by its stem
    ("runs" for "running", with ``LEXICAL_SEARCH_LANGUAGE=english``) or by a
    folded accent ("café" for "cafe") is found again.
    """
    token = _fold(token)
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            stem = token[: -len(suffix)]
            if suffix in ("ies", "ied"):
                return stem + "y"
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]  # "running" -> "runn" -> "run"
            return stem
    return token


def _densest(passages: list[str], wanted: set[str], normalize) -> str | None:
    """The passage with the most hits of ``wanted``, the first one on a tie; None without any."""
    best, best_hits = None, 0
    for passage in passages:
        hits = sum(1 for token in tokenize_words(passage) if normalize(token) in wanted)
        if hits > best_hits:
            best, best_hits = passage, hits
    return best


def best_passage(text: str, query_tokens: list[str], max_tokens: int = SNIPPET_TOKENS) -> str:
    """The passage of ``text`` with the most query-term hits; the first one on a tie.

    Exact tokens first. When no passage has one, the backend index matched
    the document some other way (a stem, a folded accent), so the tokens are
    compared loosely before falling back to the first passage.
    """
    passages = split_passages(text or "", max_tokens=max_tokens, pack=False)
    if not passages:
        return ""
    wanted = set(query_tokens)
    if not wanted:
        return passages[0]
    best = _densest(passages, wanted, lambda token: token)
    if best is None:
        best = _densest(passages, {_loose(token) for token in wanted}, _loose)
    return best if best is not None else passages[0]


class BM25DocumentsRetriever(BM25ChunksRetriever):
    """Rank cached documents with BM25 and return the best passage of each.

    ``node_name`` keeps the documents of those node sets only
    (``node_name_filter_operator`` "OR": any of them, "AND": all of them), as
    the graph searches do. On a backend index the filter runs on the best
    ``top_k * 10`` rows (at most 1000) of the whole collection, so a set whose
    documents all rank below that returns fewer hits than ``top_k``.
    """

    node_type = "CachedDocument"
    collection_name = "CachedDocument_text"
    # Conflicts hang off chunks, not off the cached documents.
    attaches_conflicts = False

    def __init__(
        self,
        top_k: int = 15,
        with_scores: bool = False,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
        **kwargs: Any,
    ):
        super().__init__(top_k=top_k, with_scores=with_scores, **kwargs)
        self.node_name = list(node_name) if node_name else None
        self.node_name_filter_operator = node_name_filter_operator

    def _keeps(self, payload: dict) -> bool:
        return payload_matches_node_filter(payload, self.node_name, self.node_name_filter_operator)

    def _native_limit(self) -> int:
        if not self.node_name:
            return self.top_k
        return min(max(self.top_k * _FILTERED_FETCH_FACTOR, self.top_k), _FILTERED_FETCH_MAX)

    async def _collect_hits(self, query: str) -> list[tuple[dict, float]]:
        hits = await super()._collect_hits(query)
        query_tokens = self._tokenizer(query)
        documents: list[tuple[dict, float]] = []
        for payload, score in hits:
            # A document that shares no term with the query is no hit: the
            # backend indexes never return one, the in-memory scorer ranks it 0.
            if score <= 0 or not self._keeps(payload):
                continue
            text = payload.get("text") or ""
            document = {key: payload[key] for key in _DOCUMENT_KEYS if key in payload}
            # The graph node and pgvector's payload call it name; LanceDB's
            # index rows carry it as document_name.
            document["name"] = payload.get("name") or payload.get("document_name")
            document["text"] = best_passage(text, query_tokens)
            document["passage_count"] = len(
                split_passages(text, max_tokens=SNIPPET_TOKENS, pack=False)
            )
            documents.append((document, score))
            if len(documents) == self.top_k:
                break
        return documents

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Any
    ) -> list[str] | list[dict]:
        """The ranked documents themselves; no LLM answer is generated."""
        if not retrieved_objects:
            return retrieved_objects
        if self.with_scores:
            return [(public_chunk_payload(payload), score) for payload, score in retrieved_objects]
        return [public_chunk_payload(payload) for payload in retrieved_objects]
