"""BM25 over the cached documents: the search that answers first (SDK-397).

``DOCUMENTS_LEXICAL`` is served by the caching pipeline's output: one
``CachedDocument`` per document, written before anything is chunked or
embedded. The ranking is the same Okapi BM25 the chunk retriever uses, either
in the vector store's lexical index (``CachedDocument_text`` on backends with
``supports_lexical_search``) or in memory over the ``CachedDocument`` graph
nodes. A hit is one document; its ``text`` is the passage with the most query
terms, never the whole document, so a result stays readable and bounded.
"""

from typing import Any

from cognee.infrastructure.databases.vector.lexical import split_passages, tokenize_words
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.utils.conflict_context import public_chunk_payload

# The snippet window: about a paragraph and a half.
SNIPPET_TOKENS = 220

# Payload keys a document hit keeps; ``text`` is replaced by the snippet.
_DOCUMENT_KEYS = ("id", "name", "document_id", "mime_type", "char_count", "truncated")


def best_passage(text: str, query_tokens: list[str], max_tokens: int = SNIPPET_TOKENS) -> str:
    """The passage of ``text`` with the most query-term hits; the first one on a tie."""
    passages = split_passages(text or "", max_tokens=max_tokens, pack=False)
    if not passages:
        return ""
    wanted = set(query_tokens)
    if not wanted:
        return passages[0]
    best, best_hits = passages[0], -1
    for passage in passages:
        hits = sum(1 for token in tokenize_words(passage) if token in wanted)
        if hits > best_hits:
            best, best_hits = passage, hits
    return best


class BM25DocumentsRetriever(BM25ChunksRetriever):
    """Rank cached documents with BM25 and return the best passage of each."""

    node_type = "CachedDocument"
    collection_name = "CachedDocument_text"
    # Conflicts hang off chunks, not off the cached documents.
    attaches_conflicts = False

    async def _collect_hits(self, query: str) -> list[tuple[dict, float]]:
        hits = await super()._collect_hits(query)
        query_tokens = self._tokenizer(query)
        documents: list[tuple[dict, float]] = []
        for payload, score in hits:
            text = payload.get("text") or ""
            document = {key: payload.get(key) for key in _DOCUMENT_KEYS if key in payload}
            document["text"] = best_passage(text, query_tokens)
            document["passage_count"] = len(
                split_passages(text, max_tokens=SNIPPET_TOKENS, pack=False)
            )
            documents.append((document, score))
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
