# import pdfplumber
from uuid import uuid4, uuid5, NAMESPACE_OID
from typing import Optional
from pypdf import PdfReader as pypdf_PdfReader
from cognee.modules.data.chunking import chunk_by_paragraph
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from .Document import Document

class PdfReader():
    # file: pdfplumber.PDF
    file = None

    def __init__(self, id: str, file_path: str):
        # self.file = pdfplumber.open(file_path)
        self.id = id
        self.file = pypdf_PdfReader(file_path)

    def read(self, max_chunk_size: Optional[int] = 1024):
        cut_chunks = []
        chunk_index = 0
        chunk_sum_size = 0

        for page in self.file.pages:
            page_text = page.extract_text()

            for chunk_data in chunk_by_paragraph(page_text, max_chunk_size, batch_paragraphs = True):
                if chunk_sum_size + chunk_data["word_count"] <= max_chunk_size:
                    cut_chunks.append(chunk_data)
                    chunk_sum_size += chunk_data["word_count"]
                else:
                    if len(cut_chunks) == 0:
                        yield DocumentChunk(
                            text = chunk_data["text"],
                            word_count = chunk_data["word_count"],
                            document_id = str(self.id),
                            chunk_id = str(chunk_data["chunk_id"]),
                            chunk_index = chunk_index,
                            cut_type = chunk_data["cut_type"],
                        )
                        cut_chunks = []
                        chunk_sum_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in cut_chunks)
                        yield DocumentChunk(
                            text = chunk_text,
                            word_count = chunk_sum_size,
                            document_id = str(self.id),
                            chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                            chunk_index = chunk_index,
                            cut_type = cut_chunks[len(cut_chunks) - 1]["cut_type"],
                        )
                        cut_chunks = [chunk_data]
                        chunk_sum_size = chunk_data["word_count"]

                    chunk_index += 1

        if len(cut_chunks) > 0:
            yield DocumentChunk(
                text = " ".join(chunk["text"] for chunk in cut_chunks),
                word_count = chunk_sum_size,
                document_id = str(self.id),
                chunk_id = str(uuid5(NAMESPACE_OID, f"{str(self.id)}-{chunk_index}")),
                chunk_index = chunk_index,
                cut_type = cut_chunks[len(cut_chunks) - 1]["cut_type"],
            )

        # self.file.close()
        self.file.stream.close()

class PdfDocument(Document):
    type: str = "pdf"
    title: str
    file_path: str

    def __init__(self, title: str, file_path: str):
        self.id = uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

    def get_reader(self) -> PdfReader:
        return PdfReader(self.id, self.file_path)

    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            file_path = self.file_path,
        )
