from cognee.shared.logging_utils import get_logger
from os.path import basename
from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from .models.DocumentChunk import DocumentChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cognee.infrastructure.databases.vector import get_vector_engine_async

logger = get_logger()


class LangchainChunker(Chunker):
    """
    A Chunker that splits text into chunks using Langchain's RecursiveCharacterTextSplitter.

    The chunker will split the text into chunks of approximately the given size, but will not split
    a chunk if the split would result in a chunk with fewer than the given overlap tokens.
    """

    def __init__(
        self,
        document,
        get_text: callable,
        max_chunk_size: int,
        chunk_size: int = 1024,
        chunk_overlap=10,
    ):
        super().__init__(document, get_text, max_chunk_size)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(text.split()),
        )

    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        # Resolve the embedding engine once — it's the same for every chunk, so
        # resolving it per chunk inside the loops just adds await/lookup overhead.
        embedding_engine = (await get_vector_engine_async()).embedding_engine
        async for content_text in self.get_text():
            for chunk in self.splitter.split_text(content_text):
                token_count = embedding_engine.tokenizer.count_tokens(chunk)
                if token_count <= self.max_chunk_size:
                    yield DocumentChunk(
                        id=uuid5(NAMESPACE_OID, chunk),
                        text=chunk,
                        chunk_size=token_count,
                        is_part_of=self.document,
                        chunk_index=self.chunk_index,
                        cut_type="missing",
                        contains=[],
                        importance_weight=self.document.importance_weight,
                        document_id=document_id,
                        document_name=document_name,
                        metadata={
                            "index_fields": ["text"],
                        },
                    )
                    self.chunk_index += 1
                else:
                    raise ValueError(
                        f"Chunk of {token_count} tokens is larger than the maximum of {self.max_chunk_size} tokens. Please reduce chunk_size in RecursiveCharacterTextSplitter."
                    )
