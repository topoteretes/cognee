"""The chunk ``external_metadata`` JSON text round-trips through a real LanceDB table.

LanceDB maps the payload model onto an Arrow struct and has no Arrow type for a
free-form dict, so the field is text end to end. This pins that a DocumentChunk
carrying metadata lands in ``DocumentChunk_text`` with the text intact, and that
a chunk without it reads back as None rather than failing the write.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import Document

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema,
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

STORED = '{"created_at": "2024-01-15", "source_id": "doc-7"}'


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _chunk(document: Document, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=2,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
        document_id=str(document.id),
        document_name=document.name,
    )


def _document(external_metadata: str | None) -> Document:
    return Document(
        id=uuid4(),
        name="doc.txt",
        raw_data_location="/tmp/doc.txt",
        external_metadata=external_metadata,
        mime_type="text/plain",
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_chunk_external_metadata_round_trips_through_lancedb(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )
    tagged = _chunk(_document(STORED), "tagged chunk")
    plain = _chunk(_document(None), "plain chunk")

    # Create the table first, as the storage task does before indexing.
    await adapter.create_collection("DocumentChunk_text", IndexSchema)
    await adapter.index_data_points("DocumentChunk", "text", [tagged, plain])

    results = await adapter.search(
        "DocumentChunk_text", query_vector=[0.1, 0.2, 0.3], limit=5, include_payload=True
    )
    payloads = {str(result.id): result.payload for result in results}
    assert payloads[str(tagged.id)]["external_metadata"] == STORED
    assert payloads[str(tagged.id)]["text"] == "tagged chunk"
    assert payloads[str(plain.id)]["external_metadata"] is None
