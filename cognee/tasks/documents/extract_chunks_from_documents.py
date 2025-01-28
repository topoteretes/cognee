from typing import Optional, AsyncGenerator

from cognee.modules.data.processing.document_types.Document import Document


async def extract_chunks_from_documents(
    documents: list[Document],
    chunk_size: int = 1024,
    chunker="text_chunker",
    max_tokens: Optional[int] = None,
) -> AsyncGenerator:
    """
    Extracts chunks of data from a list of documents based on the specified chunking parameters.

    Notes:
        - The `read` method of the `Document` class must be implemented to support the chunking operation.
        - The `chunker` parameter determines the chunking logic and should align with the document type.
    """
    for document in documents:
        document_token_count = 0
        for document_chunk in document.read(
            chunk_size=chunk_size, chunker=chunker, max_tokens=max_tokens
        ):
            document_token_count += document_chunk.token_count
            yield document_chunk
        document.token_count = document_token_count
