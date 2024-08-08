from uuid import UUID, uuid5, NAMESPACE_OID
from typing import Optional

from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from cognee.tasks.chunking.chunking_registry import get_chunking_function
from .Document import Document

class TextReader():
    id: UUID
    file_path: str
    chunking_strategy:str

    def __init__(self, id: UUID, file_path: str, chunking_strategy:str="paragraph"):
        self.id = id
        self.file_path = file_path
        self.chunking_strategy = chunking_strategy
        self.chunking_function = get_chunking_function(chunking_strategy)



    def get_number_of_pages(self):
        num_pages = 1 # Pure text is not formatted
        return num_pages

    def read(self, max_chunk_size: Optional[int] = 1024):
        chunk_index = 0
        chunk_size = 0
        chunked_pages = []
        paragraph_chunks = []

        def read_text_chunks(file_path):
            with open(file_path, mode = "r", encoding = "utf-8") as file:
                while True:
                    text = file.read(1024)

                    if len(text.strip()) == 0:
                        break

                    yield text


        page_index = 0

        for page_text in read_text_chunks(self.file_path):
            chunked_pages.append(page_index)
            page_index += 1

            for chunk_data in self.chunking_function(page_text, max_chunk_size, batch_paragraphs = True):
                if chunk_size + chunk_data["word_count"] <= max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    chunk_size += chunk_data["word_count"]
                else:
                    if len(paragraph_chunks) == 0:
                        yield DocumentChunk(
                            text = chunk_data["text"],
                            word_count = chunk_data["word_count"],
                            document_id = str(self.id),
                            chunk_id = str(chunk_data["chunk_id"]),
                            chunk_index = chunk_index,
                            cut_type = chunk_data["cut_type"],
                            pages = [page_index],
                        )
                        paragraph_chunks = []
                        chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        yield DocumentChunk(
                            text = chunk_text,
                            word_count = chunk_size,
                            document_id = str(self.id),
                            chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                            chunk_index = chunk_index,
                            cut_type = paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                            pages = chunked_pages,
                        )
                        chunked_pages = [page_index]
                        paragraph_chunks = [chunk_data]
                        chunk_size = chunk_data["word_count"]

                    chunk_index += 1

        if len(paragraph_chunks) > 0:
            yield DocumentChunk(
                text = " ".join(chunk["text"] for chunk in paragraph_chunks),
                word_count = chunk_size,
                document_id = str(self.id),
                chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                chunk_index = chunk_index,
                cut_type = paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                pages = chunked_pages,
            )

class TextDocument(Document):
    type: str = "text"
    title: str
    num_pages: int
    file_path: str

    def __init__(self, id: UUID, title: str, file_path: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

        reader = TextReader(self.id, self.file_path)
        self.num_pages = reader.get_number_of_pages()

    def get_reader(self) -> TextReader:
        reader = TextReader(self.id, self.file_path)
        return reader

    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            num_pages = self.num_pages,
            file_path = self.file_path,
        )
