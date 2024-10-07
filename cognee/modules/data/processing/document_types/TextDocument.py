from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class TextDocument(Document):
    type: str = "text"
    title: str
    raw_data_location: str

    def __init__(self, id: UUID, title: str, raw_data_location: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.raw_data_location = raw_data_location

    def read(self, chunk_size: int):
        def get_text():
            with open(self.raw_data_location, mode = "r", encoding = "utf-8") as file:
                while True:
                    text = file.read(1024)

                    if len(text.strip()) == 0:
                        break

                    yield text


        chunker = TextChunker(self.id,chunk_size = chunk_size, get_text = get_text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id = str(self.id),
            type = self.type,
            title = self.title,
            raw_data_location = self.raw_data_location,
        )
