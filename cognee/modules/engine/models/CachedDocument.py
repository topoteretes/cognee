"""A document's text as the caching pipeline stores it (SDK-397).

The caching pipeline is the first of the three runs ``cognify()`` makes. It
writes one ``CachedDocument`` per document before anything is chunked or
embedded, so the dataset answers keyword (BM25) searches seconds after
``remember()``; the Indexing and Cognify runs then add vector and graph
search on top. The node is the document's anchor in the graph (provenance,
``forget()``, rollback); the text is a lexical field, so on a backend with a
lexical index it lives in the vector store's ``CachedDocument_text`` rows and
their BM25 index and stays off the node (``vector_only_fields``). Without such
a backend the text stays on the node and the lexical retriever scores it in
memory.
"""

from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine import DataPoint

# Longer texts are cut and flagged: the cache serves keyword search, not
# archival, and a graph property of several megabytes serves nobody.
CACHED_DOCUMENT_MAX_CHARS = 2_000_000


def cached_document_id(document_id: UUID | str) -> UUID:
    """The cache node's id for a document: stable, so re-caching upserts."""
    return uuid5(NAMESPACE_OID, f"CachedDocument:{document_id}")


class CachedDocument(DataPoint):
    name: str
    text: str
    # The document (and data item) this cache entry belongs to.
    document_id: str
    mime_type: str | None = None
    char_count: int = 0
    truncated: bool = False
    metadata: dict = {
        "index_fields": [],
        "lexical_fields": ["text"],
        "vector_only_fields": ["text"],
    }
