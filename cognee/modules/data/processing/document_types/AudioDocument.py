from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class AudioDocument(Document):
    type: str = "audio"
    title: str
    raw_data_location: str
    chunking_strategy: str

    def __init__(self, id: UUID, title: str, raw_data_location: str, chunking_strategy:str="paragraph"):
        self.id = id or uuid5(NAMESPACE_OID, title)
        self.title = title
        self.raw_data_location = raw_data_location
        self.chunking_strategy = chunking_strategy

    def read(self, chunk_size: int):
        # Transcribe the audio file
        result = get_llm_client().create_transcript(self.raw_data_location)
        text = result.text

        chunker = TextChunker(self.id, chunk_size = chunk_size, get_text = lambda: text)

        yield from chunker.read()


    def to_dict(self) -> dict:
        return dict(
            id=str(self.id),
            type=self.type,
            title=self.title,
            raw_data_location=self.raw_data_location,
        )
