from uuid import UUID
from sqlalchemy import select
from typing import AsyncGenerator

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.Chunker import Chunker
from cognee.tasks.documents.exceptions import InvalidChunkSizeError, InvalidChunkerError


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
    max_chunk_size: int,
    chunker: Chunker = TextChunker,
) -> AsyncGenerator:
    """
    Extracts chunks of data from a list of documents based on the specified chunking parameters.

    Notes:
        - The `read` method of the `Document` class must be implemented to support the chunking operation.
        - The `chunker` parameter determines the chunking logic and should align with the document type.
    """
    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        raise InvalidChunkSizeError(max_chunk_size)
    if not isinstance(chunker, type):
        raise InvalidChunkerError()
    if not hasattr(chunker, "read"):
        raise InvalidChunkerError()

    for document in documents:
        document_token_count = 0

        async for document_chunk in document.read(
            max_chunk_size=max_chunk_size, chunker_cls=chunker
        ):
            document_token_count += document_chunk.chunk_size
            document_chunk.belongs_to_set = document.belongs_to_set
            yield document_chunk

        await update_document_token_count(document.id, document_token_count)
