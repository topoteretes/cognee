from typing import Optional

from cognee.modules.data.processing.document_types.Document import Document


async def extract_chunks_from_documents(
    documents: list[Document],
    chunk_size: int = 1024,
    chunker="text_chunker",
    max_tokens: Optional[int] = None,
):
    for document in documents:
        for document_chunk in document.read(
            chunk_size=chunk_size, chunker=chunker, max_tokens=max_tokens
        ):
            yield document_chunk
