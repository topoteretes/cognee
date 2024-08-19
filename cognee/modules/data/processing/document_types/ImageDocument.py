from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.chunking.TextChunker import TextChunker
from .Document import Document


class ImageDocument(Document):
    type: str = "image"
    title: str
    file_path: str

    def __init__(self, id: UUID, title: str, file_path: str):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.file_path = file_path

    def read(self):
        # Transcribe the image file
        result = get_llm_client().transcribe_image(self.file_path)
        text = result.choices[0].message.content

        chunker = TextChunker(self.id, get_text = lambda: text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id=str(self.id),
            type=self.type,
            title=self.title,
            file_path=self.file_path,
        )
