from uuid import uuid5, NAMESPACE_OID

from .models.DocumentChunk import DocumentChunk
from cognee.tasks.chunks import chunk_by_paragraph

class TextChunker():
    document = None
    max_chunk_size: int

    chunk_index = 0
    chunk_size = 0

    def __init__(self, document, get_text: callable, chunk_size: int = 1024):
        self.document = document
        self.max_chunk_size = chunk_size
        self.get_text = get_text

    def read(self):
        paragraph_chunks = []
        for content_text in self.get_text():
            for chunk_data in chunk_by_paragraph(
                content_text,
                self.max_chunk_size,
                batch_paragraphs = True,
            ):
                if self.chunk_size + chunk_data["word_count"] <= self.max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    self.chunk_size += chunk_data["word_count"]
                else:
                    if len(paragraph_chunks) == 0:
                        yield DocumentChunk(
                            id = chunk_data["chunk_id"],
                            text = chunk_data["text"],
                            word_count = chunk_data["word_count"],
                            is_part_of = self.document,
                            chunk_index = self.chunk_index,
                            cut_type = chunk_data["cut_type"],
                            contains = [],
                            _metadata = {
                                "index_fields": ["text"],
                                "metadata_id": self.document.metadata_id
                            }
                        )
                        paragraph_chunks = []
                        self.chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        try:
                            yield DocumentChunk(
                                id = uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"),
                                text = chunk_text,
                                word_count = self.chunk_size,
                                is_part_of = self.document,
                                chunk_index = self.chunk_index,
                                cut_type = paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                                contains = [],
                                _metadata = {
                                    "index_fields": ["text"],
                                    "metadata_id": self.document.metadata_id
                                }
                            )
                        except Exception as e:
                            print(e)
                        paragraph_chunks = [chunk_data]
                        self.chunk_size = chunk_data["word_count"]

                    self.chunk_index += 1

        if len(paragraph_chunks) > 0:
            try:
                yield DocumentChunk(
                    id = uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"),
                    text = " ".join(chunk["text"] for chunk in paragraph_chunks),
                    word_count = self.chunk_size,
                    is_part_of = self.document,
                    chunk_index = self.chunk_index,
                    cut_type = paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                    contains = [],
                    _metadata = {
                        "index_fields": ["text"],
                        "metadata_id": self.document.metadata_id
                    }
                )
            except Exception as e:
                print(e)
