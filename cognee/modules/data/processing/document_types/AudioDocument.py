from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Optional

from cognee.infrastructure.llm.get_llm_client import get_llm_client

from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document

from cognee.tasks.chunking.chunking_registry import get_chunking_function

class AudioReader:
    id: UUID
    file_path: str
    chunking_strategy:str

    def __init__(self, id: UUID, file_path: str, chunking_strategy:str = "paragraph"):
        self.id = id
        self.file_path = file_path
        self.llm_client = get_llm_client()  # You can choose different models like "tiny", "base", "small", etc.
        self.chunking_function = get_chunking_function(chunking_strategy)

    def read(self, max_chunk_size: Optional[int] = 1024):
        chunk_index = 0
        chunk_size = 0
        chunked_pages = []
        paragraph_chunks = []

        # Transcribe the audio file
        result = self.llm_client.create_transcript(self.file_path)
        text = result.text

        # Simulate reading text in chunks as done in TextReader
        def read_text_chunks(text, chunk_size):
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]

        page_index = 0

        for page_text in read_text_chunks(text, max_chunk_size):
            chunked_pages.append(page_index)
            page_index += 1

            for chunk_data in self.chunking_function(page_text, max_chunk_size, batch_paragraphs=True):
                if chunk_size + chunk_data["word_count"] <= max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    chunk_size += chunk_data["word_count"]
                else:
                    if len(paragraph_chunks) == 0:
                        yield DocumentChunk(
                            text=chunk_data["text"],
                            word_count=chunk_data["word_count"],
                            document_id=str(self.id),
                            chunk_id=str(chunk_data["chunk_id"]),
                            chunk_index=chunk_index,
                            cut_type=chunk_data["cut_type"],
                            pages=[page_index],
                        )
                        paragraph_chunks = []
                        chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        yield DocumentChunk(
                            text=chunk_text,
                            word_count=chunk_size,
                            document_id=str(self.id),
                            chunk_id=str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                            chunk_index=chunk_index,
                            cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                            pages=chunked_pages,
                        )
                        chunked_pages = [page_index]
                        paragraph_chunks = [chunk_data]
                        chunk_size = chunk_data["word_count"]

                    chunk_index += 1

        if len(paragraph_chunks) > 0:
            yield DocumentChunk(
                text=" ".join(chunk["text"] for chunk in paragraph_chunks),
                word_count=chunk_size,
                document_id=str(self.id),
                chunk_id=str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                chunk_index=chunk_index,
                cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                pages=chunked_pages,
            )

class AudioDocument(Document):
    type: str = "audio"
    title: str
    file_path: str
    chunking_strategy:str

    def __init__(self, id: UUID, title: str, file_path: str, chunking_strategy:str="paragraph"):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path
        self.chunking_strategy = chunking_strategy

    def get_reader(self) -> AudioReader:
        reader = AudioReader(self.id, self.file_path, self.chunking_strategy)
        return reader

    def to_dict(self) -> dict:
        return dict(
            id=str(self.id),
            type=self.type,
            title=self.title,
            file_path=self.file_path,
        )


# if __name__ == "__main__":
#     # Sample usage of AudioDocument
#     audio_document = AudioDocument("sample_audio", "/Users/vasa/Projects/cognee/cognee/modules/data/processing/document_types/preamble10.wav")
#     audio_reader = audio_document.get_reader()
#     for chunk in audio_reader.read():
#         print(chunk.text)
#         print(chunk.word_count)
#         print(chunk.document_id)
#         print(chunk.chunk_id)
#         print(chunk.chunk_index)
#         print(chunk.cut_type)
#         print(chunk.pages)
#         print("----")
