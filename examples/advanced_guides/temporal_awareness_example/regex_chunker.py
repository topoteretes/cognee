import re
from os.path import basename

from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.tasks.chunks import chunk_by_paragraph


class RegexChunker(Chunker):
    """Split on a regex first; fall back to paragraph chunking for oversized parts.

    The temporal filter works per chunk, and a period's owner, ``begins_at``, and
    ``ends_at`` must share a chunk — one blank-line section per chunk keeps that true.
    """

    chunker_id = "regex_chunker_v1"
    split_pattern = r"\n\s*\n"

    def _split_sections(self, text: str) -> list[str]:
        return [section for section in re.split(self.split_pattern, text) if section.strip()]

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        hash_occurrences: dict[str, int] = {}

        def chunk_identity(text: str):
            content_hash = chunk_content_hash(text)
            occurrence = hash_occurrences.get(content_hash, 0)
            hash_occurrences[content_hash] = occurrence + 1
            return content_chunk_id(document_id, content_hash, occurrence), content_hash

        async for content_text in self.get_text():
            for section in self._split_sections(content_text):
                for chunk_data in chunk_by_paragraph(section, self.max_chunk_size):
                    chunk_id, content_hash = chunk_identity(chunk_data["text"])
                    yield DocumentChunk(
                        chunker_id=self.chunker_id,
                        id=chunk_id,
                        text=chunk_data["text"],
                        chunk_size=chunk_data["chunk_size"],
                        content_hash=content_hash,
                        max_chunk_tokens=self.max_chunk_size,
                        is_part_of=self.document,
                        chunk_index=self.chunk_index,
                        cut_type=chunk_data["cut_type"],
                        contains=[],
                        importance_weight=self.document.importance_weight,
                        document_id=document_id,
                        document_name=document_name,
                        metadata={"index_fields": ["text"]},
                    )
                    self.chunk_index += 1
