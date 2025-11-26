from cognee.shared.logging_utils import get_logger


from cognee.tasks.chunks import chunk_by_row
from cognee.modules.chunking.Chunker import Chunker
from .models.DocumentChunk import DocumentChunk

logger = get_logger()


class CsvChunker(Chunker):
    async def read(self):
        async for content_text in self.get_text():
            if content_text is None:
                continue

            for chunk_data in chunk_by_row(content_text, self.max_chunk_size):
                if chunk_data["chunk_size"] <= self.max_chunk_size:
                    yield DocumentChunk(
                        id=chunk_data["chunk_id"],
                        text=chunk_data["text"],
                        chunk_size=chunk_data["chunk_size"],
                        is_part_of=self.document,
                        chunk_index=self.chunk_index,
                        cut_type=chunk_data["cut_type"],
                        contains=[],
                        metadata={
                            "index_fields": ["text"],
                        },
                    )
                    self.chunk_index += 1
                else:
                    raise ValueError(
                        f"Chunk size is larger than the maximum chunk size {self.max_chunk_size}"
                    )
