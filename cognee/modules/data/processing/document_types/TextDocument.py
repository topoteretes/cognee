from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.data.chunking.TextChunker import TextChunker
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
        def get_text():
            with open(self.file_path, mode = "r", encoding = "utf-8") as file:
                while True:
                    text = file.read(1024)

                    if len(text.strip()) == 0:
                        break

                    yield text


        chunker = TextChunker(self.id, get_text = get_text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            file_path = self.file_path,
        )
