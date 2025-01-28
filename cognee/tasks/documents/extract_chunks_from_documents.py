from typing import AsyncGenerator

from cognee.modules.data.processing.document_types.Document import Document
from sqlalchemy import select
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from uuid import UUID


async def update_document_token_count(document_id: UUID, token_count: int) -> None:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        document_data_point = (
            await session.execute(select(Data).filter(Data.id == document_id))
        ).scalar_one_or_none()

        if document_data_point:
            document_data_point.token_count = token_count
            await session.merge(document_data_point)
            await session.commit()
        else:
            raise ValueError(f"Document with id {document_id} not found.")


async def extract_chunks_from_documents(
    documents: list[Document],
    max_chunk_tokens: int,
    chunk_size: int = 1024,
    chunker="text_chunker",
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
            chunk_size=chunk_size, chunker=chunker, max_chunk_tokens=max_chunk_tokens
        ):
            document_token_count += document_chunk.token_count
            yield document_chunk

        await update_document_token_count(document.id, document_token_count)
