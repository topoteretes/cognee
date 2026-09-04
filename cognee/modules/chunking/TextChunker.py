from cognee.shared.logging_utils import get_logger
from os.path import basename

from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from .models.DocumentChunk import DocumentChunk

logger = get_logger()


class TextChunker(Chunker):
    chunker_id = "text_chunker_v1"

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        paragraph_chunks = []
        # Chunk identity is content-derived; the occurrence counter keeps two
        # identical texts in one document distinct (see chunk_id module).
        hash_occurrences: dict = {}

        def chunk_identity(text: str):
            content_hash = chunk_content_hash(text)
            occurrence = hash_occurrences.get(content_hash, 0)
            hash_occurrences[content_hash] = occurrence + 1
            return content_chunk_id(document_id, content_hash, occurrence), content_hash

        async for content_text in self.get_text():
            for chunk_data in chunk_by_paragraph(
                content_text,
                self.max_chunk_size,
                batch_paragraphs=True,
            ):
                if self.chunk_size + chunk_data["chunk_size"] <= self.max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    self.chunk_size += chunk_data["chunk_size"]
                else:
                    if len(paragraph_chunks) == 0:
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
                            metadata={
                                "index_fields": ["text"],
                            },
                        )
                        paragraph_chunks = []
                        self.chunk_size = 0
                    else:
                        chunk_text = "".join(chunk["text"] for chunk in paragraph_chunks)
                        try:
                            chunk_id, content_hash = chunk_identity(chunk_text)
                            yield DocumentChunk(
                                chunker_id=self.chunker_id,
                                id=chunk_id,
                                text=chunk_text,
                                chunk_size=self.chunk_size,
                                content_hash=content_hash,
                                max_chunk_tokens=self.max_chunk_size,
                                is_part_of=self.document,
                                chunk_index=self.chunk_index,
                                cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                                contains=[],
                                importance_weight=self.document.importance_weight,
                                document_id=document_id,
                                document_name=document_name,
                                metadata={
                                    "index_fields": ["text"],
                                },
                            )
                        except Exception as e:
                            logger.error(e)
                            raise e
                        paragraph_chunks = [chunk_data]
                        self.chunk_size = chunk_data["chunk_size"]

                    self.chunk_index += 1

        if len(paragraph_chunks) > 0:
            try:
                chunk_text = "".join(chunk["text"] for chunk in paragraph_chunks)
                chunk_id, content_hash = chunk_identity(chunk_text)
                yield DocumentChunk(
                    chunker_id=self.chunker_id,
                    id=chunk_id,
                    text=chunk_text,
                    chunk_size=self.chunk_size,
                    content_hash=content_hash,
                    max_chunk_tokens=self.max_chunk_size,
                    is_part_of=self.document,
                    chunk_index=self.chunk_index,
                    cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                    contains=[],
                    importance_weight=self.document.importance_weight,
                    document_id=document_id,
                    document_name=document_name,
                    metadata={"index_fields": ["text"]},
                )
            except Exception as e:
                logger.error(e)
                raise e
