from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.data.chunking.DocumentChunker import DocumentChunker
from .Document import Document

class TextDocument(Document):
    type: str = "text"
    title: str
    file_path: str

    def __init__(self, id: UUID, title: str, file_path: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

    def read(self):
        document_chunker = DocumentChunker(self.id)

        def read_text_chunks(file_path):
            with open(file_path, mode = "r", encoding = "utf-8") as file:
                while True:
                    text = file.read(1024)

                    if len(text.strip()) == 0:
                        break

                    yield text
        
        for text_chunk in read_text_chunks(self.file_path):
            yield from document_chunker.read(text_chunk)


    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            file_path = self.file_path,
        )
