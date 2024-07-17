from uuid import uuid5, NAMESPACE_OID
from typing import Optional, Generator

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.chunking import chunk_by_paragraph
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document

class AudioReader:
    id: str
    file_path: str

    def __init__(self, id: str, file_path: str):
        self.id = id
        self.file_path = file_path
        self.llm_client = get_llm_client()  # You can choose different models like "tiny", "base", "small", etc.

    def get_number_of_pages(self):
        # Estimate the number of pages based on audio duration (1 minute = 1 page for simplicity)
        result = self.llm_client.create_transcript(self.file_path)
        duration = result['duration'] / 60  # Duration in minutes
        return int(duration) + 1

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

            for chunk_data in chunk_by_paragraph(page_text, max_chunk_size, batch_paragraphs=True):
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

    def __init__(self, title: str, file_path: str):
        self.id = uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

        reader = AudioReader(self.id, self.file_path)

    def get_reader(self) -> AudioReader:
        reader = AudioReader(self.id, self.file_path)
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
