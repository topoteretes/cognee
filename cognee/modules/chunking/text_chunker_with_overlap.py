from cognee.shared.logging_utils import get_logger
from uuid import NAMESPACE_OID, uuid5

from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker
from .models.DocumentChunk import DocumentChunk

logger = get_logger()


class TextChunkerWithOverlap(Chunker):
    def __init__(
        self,
        document,
        get_text: callable,
        max_chunk_size: int,
        chunk_overlap_ratio: float = 0.0,
        get_chunk_data: callable = None,
    ):
        super().__init__(document, get_text, max_chunk_size)
        self._accumulated_chunk_data = []
        self._accumulated_size = 0
        self.chunk_overlap_ratio = chunk_overlap_ratio
        self.chunk_overlap = int(max_chunk_size * chunk_overlap_ratio)

        if get_chunk_data is not None:
            self.get_chunk_data = get_chunk_data
        elif chunk_overlap_ratio > 0:
            paragraph_max_size = int(0.5 * chunk_overlap_ratio * max_chunk_size)
            self.get_chunk_data = lambda text: chunk_by_paragraph(
                text, paragraph_max_size, batch_paragraphs=True
            )
        else:
            self.get_chunk_data = lambda text: chunk_by_paragraph(
                text, self.max_chunk_size, batch_paragraphs=True
            )

    def _accumulation_overflows(self, chunk_data):
        """Check if adding chunk_data would exceed max_chunk_size."""
        return self._accumulated_size + chunk_data["chunk_size"] > self.max_chunk_size

    def _accumulate_chunk_data(self, chunk_data):
        """Add chunk_data to the current accumulation."""
        self._accumulated_chunk_data.append(chunk_data)
        self._accumulated_size += chunk_data["chunk_size"]

    def _clear_accumulation(self):
        """Reset accumulation, keeping overlap chunk_data based on chunk_overlap_ratio."""
        if self.chunk_overlap == 0:
            self._accumulated_chunk_data = []
            self._accumulated_size = 0
            return

        # Keep chunk_data from the end that fit in overlap
        overlap_chunk_data = []
        overlap_size = 0

        for chunk_data in reversed(self._accumulated_chunk_data):
            if overlap_size + chunk_data["chunk_size"] <= self.chunk_overlap:
                overlap_chunk_data.insert(0, chunk_data)
                overlap_size += chunk_data["chunk_size"]
            else:
                break

        self._accumulated_chunk_data = overlap_chunk_data
        self._accumulated_size = overlap_size

    def _create_chunk(self, text, size, cut_type, chunk_id=None):
        """Create a DocumentChunk with standard metadata."""
        try:
            return DocumentChunk(
                id=chunk_id or uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"),
                text=text,
                chunk_size=size,
                is_part_of=self.document,
                chunk_index=self.chunk_index,
                cut_type=cut_type,
                contains=[],
                metadata={"index_fields": ["text"]},
            )
        except Exception as e:
            logger.error(e)
            raise e

    def _create_chunk_from_accumulation(self):
        """Create a DocumentChunk from current accumulated chunk_data."""
        chunk_text = " ".join(chunk["text"] for chunk in self._accumulated_chunk_data)
        return self._create_chunk(
            text=chunk_text,
            size=self._accumulated_size,
            cut_type=self._accumulated_chunk_data[-1]["cut_type"],
        )

    def _emit_chunk(self, chunk_data):
        """Emit a chunk when accumulation overflows."""
        if len(self._accumulated_chunk_data) > 0:
            chunk = self._create_chunk_from_accumulation()
            self._clear_accumulation()
            self._accumulate_chunk_data(chunk_data)
        else:
            # Handle single chunk_data exceeding max_chunk_size
            chunk = self._create_chunk(
                text=chunk_data["text"],
                size=chunk_data["chunk_size"],
                cut_type=chunk_data["cut_type"],
                chunk_id=chunk_data["chunk_id"],
            )

        self.chunk_index += 1
        return chunk

    async def read(self):
        async for content_text in self.get_text():
            for chunk_data in self.get_chunk_data(content_text):
                if not self._accumulation_overflows(chunk_data):
                    self._accumulate_chunk_data(chunk_data)
                    continue

                yield self._emit_chunk(chunk_data)

        if len(self._accumulated_chunk_data) == 0:
            return

        yield self._create_chunk_from_accumulation()
