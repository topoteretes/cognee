"""Unit tests for CsvChunker chunk identity.

``chunk_by_row`` derives its ``chunk_id`` from the row text alone
(``uuid5(NAMESPACE_OID, text)``). Passing that straight through as the
``DocumentChunk`` id makes identity neither document-scoped nor
occurrence-counted, so a CSV with repeated rows loses chunks to the dedup pass
in ``add_data_points`` and two documents sharing a row share one chunk node.
CsvChunker uses ``content_chunk_id(document_id, content_hash, occurrence)``
instead, the same scheme TextChunker uses.

The embedding engine is mocked the way the repo's own CSV test does it
(cognee/tests/integration/documents/CsvDocument_test.py), so these tests need
no network and no tokenizer download.
"""

import importlib
from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.CsvChunker import CsvChunker
from cognee.modules.data.processing.document_types.CsvDocument import CsvDocument
from cognee.modules.graph.utils import deduplicate_nodes_and_edges
from cognee.tests.integration.documents.AudioDocument_test import mock_get_embedding_engine

# importlib rather than `import ... as`: tasks/chunks/__init__.py re-exports the
# functions under the same names as their modules, so the dotted import resolves
# to the function and patch.object has nothing to patch.
_chunk_by_row = importlib.import_module("cognee.tasks.chunks.chunk_by_row")

REPEATED_ROW = "name: John, age: 30"
OTHER_ROW = "name: Jane, age: 25"


@pytest.fixture(autouse=True)
def _mocked_embedding_engine():
    with patch.object(_chunk_by_row, "get_embedding_engine", side_effect=mock_get_embedding_engine):
        yield


def _csv_document(name):
    return CsvDocument(
        id=uuid4(),
        name=name,
        raw_data_location=f"/test/path/{name}",
        external_metadata="",
        mime_type="text/csv",
    )


def _text_generator(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


async def _collect(document, *texts, max_chunk_size=128):
    chunker = CsvChunker(document, _text_generator(*texts), max_chunk_size)
    return [chunk async for chunk in chunker.read()]


@pytest.mark.asyncio
async def test_repeated_rows_stay_separate_chunks():
    """Two identical rows in one document are two chunks that both survive dedup."""
    document = _csv_document("dupes.csv")
    chunks = await _collect(document, f"{REPEATED_ROW}\n\n{REPEATED_ROW}\n\n{OTHER_ROW}")

    assert len(chunks) == 3, "Three rows should produce three chunks"
    assert chunks[0].text == chunks[1].text, "The first two rows carry identical text"
    assert chunks[0].id != chunks[1].id, "Identical rows must not share a chunk id"
    assert len({chunk.id for chunk in chunks}) == 3, "Every chunk id should be distinct"

    kept, _ = deduplicate_nodes_and_edges(list(chunks), [])
    assert len(kept) == 3, "Dedup by id must not drop a repeated row"
    assert [chunk.chunk_index for chunk in kept] == [0, 1, 2]


@pytest.mark.asyncio
async def test_same_row_in_two_documents_gets_different_ids():
    """A row shared by two documents must not collapse into one chunk node."""
    document_a = _csv_document("a.csv")
    document_b = _csv_document("b.csv")

    chunk_a = (await _collect(document_a, OTHER_ROW))[0]
    chunk_b = (await _collect(document_b, OTHER_ROW))[0]

    assert chunk_a.text == chunk_b.text
    assert chunk_a.id != chunk_b.id, "Chunk identity must be scoped to its document"

    kept, _ = deduplicate_nodes_and_edges([chunk_a, chunk_b], [])
    assert len(kept) == 2, "Both documents keep their own chunk node"
    assert {chunk.document_name for chunk in kept} == {"a.csv", "b.csv"}


@pytest.mark.asyncio
async def test_chunk_id_is_derived_from_document_content_and_occurrence():
    """Ids match content_chunk_id, so re-ingesting the same document is idempotent."""
    document = _csv_document("dupes.csv")
    chunks = await _collect(document, f"{REPEATED_ROW}\n\n{REPEATED_ROW}")

    content_hash = chunk_content_hash(REPEATED_ROW)
    document_id = str(document.id)
    assert [chunk.id for chunk in chunks] == [
        content_chunk_id(document_id, content_hash, 0),
        content_chunk_id(document_id, content_hash, 1),
    ]
    assert [chunk.content_hash for chunk in chunks] == [content_hash, content_hash]
    assert all(chunk.max_chunk_tokens == 128 for chunk in chunks)

    rerun = await _collect(document, f"{REPEATED_ROW}\n\n{REPEATED_ROW}")
    assert [chunk.id for chunk in rerun] == [chunk.id for chunk in chunks]
