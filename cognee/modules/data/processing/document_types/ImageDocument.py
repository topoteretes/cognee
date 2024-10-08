from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document


class ImageDocument(Document):
    type: str = "image"
    title: str
    raw_data_location: str

    def __init__(self, id: UUID, title: str, raw_data_location: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.raw_data_location = raw_data_location

    def read(self, chunk_size: int):
        # Transcribe the image file
        result = get_llm_client().transcribe_image(self.raw_data_location)
        text = result.choices[0].message.content

        chunker = TextChunker(self.id, chunk_size = chunk_size, get_text = lambda: text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id=str(self.id),
            type=self.type,
            title=self.title,
            raw_data_location=self.raw_data_location,
        )
