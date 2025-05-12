from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.data.processing.document_types import Document


class TextDocument(Document):
    type: str = "text"
    mime_type: str = "text/plain"
    content: str

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        def get_text():
            yield self.content

        chunker: Chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)
        yield from chunker.read()
