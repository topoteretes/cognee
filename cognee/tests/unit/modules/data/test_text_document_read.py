"""TextDocument.read must deliver the WHOLE file — probe for the truncation bug.

The old reader stopped at any read block that was entirely whitespace
(`if not text.strip(): break`). With 1,000,000-character read blocks, a long
whitespace run landing on a block boundary silently dropped everything after
it. The probe builds exactly that shape: a 1 MB block of content, a 1 MB
block of pure whitespace, then a tail with a marker — and asserts the chunker
receives the tail and the reassembled text is byte-identical to the file.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path

BLOCK = 1_000_000


def test_whitespace_block_does_not_truncate():
    asyncio.run(_scenario())


async def _scenario():
    from cognee.modules.chunking.TextChunker import TextChunker
    from cognee.modules.data.processing.document_types.TextDocument import TextDocument

    # 40-character lines so the head is EXACTLY one read block: the next
    # block is then pure whitespace, the shape the old reader broke on.
    line = "The archivist catalogued shipment 0007.\n"
    assert len(line) == 40
    head = line * (BLOCK // 40)
    assert len(head) == BLOCK
    whitespace_run = " " * BLOCK
    tail = "TAIL_MARKER: the final paragraph that the old reader silently dropped.\n"
    full_text = head + whitespace_run + tail

    path = Path(tempfile.mkdtemp(prefix="cognee_truncation_probe_")) / "doc.txt"
    path.write_text(full_text, encoding="utf-8")

    document = TextDocument(
        id=uuid.uuid4(),
        title="doc.txt",
        raw_data_location=str(path),
        name="doc",
        mime_type="text/plain",
        external_metadata="{}",
    )

    # The reader itself, byte-for-byte: a capture chunker passes the raw
    # stream pieces through untouched, isolating the unit under test.
    from types import SimpleNamespace

    class _CaptureChunker:
        def __init__(self, document, max_chunk_size, get_text):
            self.get_text = get_text

        async def read(self):
            async for piece in self.get_text():
                yield SimpleNamespace(text=piece)

    pieces = [chunk.text async for chunk in document.read(_CaptureChunker, max_chunk_size=512)]
    assert "".join(pieces) == full_text, "the reader must deliver the file byte-for-byte"
    assert len(pieces) >= 3, "the whitespace-only block must be yielded, not treated as EOF"

    # And through the real chunker: read-block boundaries must not change text.
    chunks = [chunk async for chunk in document.read(TextChunker, max_chunk_size=512)]
    reassembled = "".join(chunk.text for chunk in chunks)
    assert reassembled == full_text, "the real chunker must preserve every reader block exactly"
