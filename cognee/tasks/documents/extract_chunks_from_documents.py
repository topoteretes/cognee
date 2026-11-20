import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunker_registry import resolve_chunker, stamp_chunk
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.utils.temporal_hints import hint_lines
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents.exceptions import InvalidChunkerError, InvalidChunkSizeError

# The date-hint pass (dateparser) is pure CPU and takes ~0.2 s on a chunk dense
# with numbers and dates. Run on the event loop, a large document's chunks are
# cut and hinted with no await in between, which starves every other coroutine
# (DB handshakes, health checks) for minutes. One worker thread keeps the loop
# responsive; one is enough because the GIL serializes the work anyway, and it
# keeps concurrent documents from calling dateparser from several threads.
_HINT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="temporal-hints")


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


@task_summary("Extracted {n} chunk(s)")
async def extract_chunks_from_documents(
    documents: list[Document],
    max_chunk_size: int,
    chunker: type[Chunker] | None = None,
    temporal_hints: bool = True,
) -> AsyncGenerator:
    """
    Extracts chunks of data from a list of documents based on the specified chunking parameters.

    Notes:
        - The `read` method of the `Document` class must be implemented to support the chunking operation.
        - The `chunker` parameter determines the chunking logic and should align with the document type.
        - `temporal_hints` attaches the date hints graph extraction puts in its prompt. The
          indexing pipeline turns it off: its chunks are only stored and embedded, the hints
          are never persisted, and the dateparser pass costs more than cutting the chunks.
    """
    chunker = resolve_chunker(chunker)
    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        raise InvalidChunkSizeError(max_chunk_size)
    if not isinstance(chunker, type):
        raise InvalidChunkerError()
    if not hasattr(chunker, "read"):
        raise InvalidChunkerError()

    for document in documents:
        document_token_count = 0
        # This is the one place that sees a document's chunks in order, so the
        # date hints are computed here and travel with each chunk: the rolling
        # base is local to this loop and never crosses into another document
        # (the same pass ``document_temporal_hints`` runs over a text list).
        temporal_base = None

        async for document_chunk in document.read(
            max_chunk_size=max_chunk_size, chunker_cls=chunker
        ):
            stamp_chunk(chunker, document_chunk)
            document_token_count += document_chunk.chunk_size
            document_chunk.belongs_to_set = document.belongs_to_set
            if temporal_hints:
                (
                    document_chunk._temporal_hints,
                    temporal_base,
                ) = await asyncio.get_running_loop().run_in_executor(
                    _HINT_EXECUTOR, hint_lines, document_chunk.text, temporal_base
                )
            else:
                # The chunker itself never awaits; give the loop a turn per chunk.
                await asyncio.sleep(0)
            yield document_chunk

        await update_document_token_count(document.id, document_token_count)
