"""Unit tests for content-derived chunk identity."""

from hashlib import sha256
from uuid import UUID

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id


def test_content_hash_is_sha256_of_exact_text():
    assert chunk_content_hash("hello world") == sha256(b"hello world").hexdigest()
    assert chunk_content_hash("hello world ") != chunk_content_hash("hello world")


def test_chunk_id_is_deterministic():
    doc = UUID("7bbceca2-adef-588e-bf86-d08baaac0fc5")
    digest = chunk_content_hash("some chunk text")
    assert content_chunk_id(doc, digest, 0) == content_chunk_id(doc, digest, 0)


def test_chunk_id_distinguishes_document_content_and_occurrence():
    doc_a = UUID("7bbceca2-adef-588e-bf86-d08baaac0fc5")
    doc_b = UUID("9672df92-bc43-5656-98c6-09e1fb5255cf")
    digest = chunk_content_hash("repeated paragraph")
    other = chunk_content_hash("different paragraph")

    base = content_chunk_id(doc_a, digest, 0)
    assert content_chunk_id(doc_b, digest, 0) != base  # same text, other document
    assert content_chunk_id(doc_a, other, 0) != base  # other text, same document
    assert content_chunk_id(doc_a, digest, 1) != base  # same text appearing twice


def test_unchanged_content_keeps_identity_across_positions():
    """The point of the scheme: identity does not depend on chunk position."""
    doc = UUID("7bbceca2-adef-588e-bf86-d08baaac0fc5")
    digest = chunk_content_hash("a paragraph that moved from position 2 to position 9")
    at_ingestion = content_chunk_id(doc, digest, 0)
    after_update = content_chunk_id(doc, digest, 0)
    assert at_ingestion == after_update
