from uuid import UUID, uuid5, NAMESPACE_OID
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.data.chunking.DocumentChunker import DocumentChunker
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
#     audio_document = ImageDocument("sample_audio", "/Users/vasa/Projects/cognee/assets/architecture.png")
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
