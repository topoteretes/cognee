from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.chunking.TextChunker import TextChunker
from .Document import Document

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

    def read(self):
        # Transcribe the audio file
        result = get_llm_client().create_transcript(self.file_path)
        text = result.text

        chunker = TextChunker(self.id, get_text = lambda: text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id=str(self.id),
            type=self.type,
            title=self.title,
            file_path=self.file_path,
        )
