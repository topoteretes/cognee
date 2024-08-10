from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Optional

from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from cognee.tasks.chunking import chunk_by_paragraph

class DocumentChunker():
    id: UUID
    chunk_index = 0
    chunk_size = 0
    paragraph_chunks = []

    def __init__(self, id: UUID):
        self.id = id

    def read(self, content_text: str, max_chunk_size: Optional[int] = 1024):
        for chunk_data in chunk_by_paragraph(content_text, max_chunk_size, batch_paragraphs = True):
            if self.chunk_size + chunk_data["word_count"] <= max_chunk_size:
                self.paragraph_chunks.append(chunk_data)
                self.chunk_size += chunk_data["word_count"]
            else:
                if len(self.paragraph_chunks) == 0:
                    yield DocumentChunk(
                        text = chunk_data["text"],
                        word_count = chunk_data["word_count"],
                        document_id = str(self.id),
                        chunk_id = str(chunk_data["chunk_id"]),
                        chunk_index = self.chunk_index,
                        cut_type = chunk_data["cut_type"],
                    )
                    self.paragraph_chunks = []
                    self.chunk_size = 0
                else:
                    chunk_text = " ".join(chunk["text"] for chunk in self.paragraph_chunks)
                    yield DocumentChunk(
                        text = chunk_text,
                        word_count = self.chunk_size,
                        document_id = str(self.id),
                        chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{self.chunk_index}")),
                        chunk_index = self.chunk_index,
                        cut_type = self.paragraph_chunks[len(self.paragraph_chunks) - 1]["cut_type"],
                    )
                    self.paragraph_chunks = [chunk_data]
                    self.chunk_size = chunk_data["word_count"]

                self.chunk_index += 1

        if len(self.paragraph_chunks) > 0:
            yield DocumentChunk(
                text = " ".join(chunk["text"] for chunk in self.paragraph_chunks),
                word_count = self.chunk_size,
                document_id = str(self.id),
                chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{self.chunk_index}")),
                chunk_index = self.chunk_index,
                cut_type = self.paragraph_chunks[len(self.paragraph_chunks) - 1]["cut_type"],
            )
