from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.chunking.DocumentChunker import DocumentChunker
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

        document_chunker = DocumentChunker(self.id)
        yield from document_chunker.read(text)

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
