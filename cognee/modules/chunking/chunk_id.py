"""Content-derived chunk identity.

Chunk ids were previously positional (``uuid5(doc_id-chunk_index)``), which
made every id downstream of an edit shift or collide once chunk indices moved.
Identity is now derived from the chunk's content hash instead: an unchanged
chunk keeps its id no matter where it sits, and re-ingesting identical content
is idempotent. The occurrence counter disambiguates identical texts appearing
more than once in the same document (two equal paragraphs stay two nodes).
"""

from hashlib import sha256
from uuid import NAMESPACE_OID, UUID, uuid5


def chunk_content_hash(text: str) -> str:
    """Stable hex digest of the chunk's exact text."""
    return sha256(text.encode("utf-8")).hexdigest()


def content_chunk_id(document_id, content_hash: str, occurrence: int) -> UUID:
    """Deterministic chunk id from (document, content, nth occurrence)."""
    return uuid5(NAMESPACE_OID, f"{document_id}:{content_hash}:{occurrence}")
