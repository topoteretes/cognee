"""The caching pipeline's write: a document's text, searchable by keyword (SDK-397).

``cognify()`` runs three pipelines per dataset. The first, ``caching_pipeline``
(``classify_documents`` → this task), reads each document's processed text —
the file ingestion stored, already extracted from PDFs and the like — and
stores it as one ``CachedDocument``. No chunker, no embedding, no LLM: the
dataset answers ``DOCUMENTS_LEXICAL`` (BM25) searches as soon as this run
completes, and the Indexing and Cognify runs add vector and graph search on
top. A document whose stored text is not UTF-8 is skipped with a warning
(reading it again cannot succeed); any other read error fails the item, so it
is not stamped and the next ``cognify()`` caches it. Either way the later
runs still process it.
"""

from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import (
    CACHED_DOCUMENT_MAX_CHARS,
    CachedDocument,
    cached_document_id,
)
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points

logger = get_logger("cache_documents")


async def read_document_text(document: Document) -> str:
    """The document's processed text as ingestion stored it."""
    async with open_data_file(document.raw_data_location, mode="r", encoding="utf-8") as file:
        return file.read()


def build_cached_document(document: Document, text: str) -> CachedDocument:
    truncated = len(text) > CACHED_DOCUMENT_MAX_CHARS
    return CachedDocument(
        id=cached_document_id(document.id),
        name=document.name,
        text=text[:CACHED_DOCUMENT_MAX_CHARS] if truncated else text,
        document_id=str(document.id),
        mime_type=document.mime_type,
        char_count=len(text),
        truncated=truncated,
        belongs_to_set=document.belongs_to_set,
    )


def _entry_for(document: Document, text: str) -> CachedDocument | None:
    """The cache entry for ``text``, or None for a blank one; long texts are cut and logged."""
    if not text.strip():
        return None
    entry = build_cached_document(document, text)
    if entry.truncated:
        logger.info(
            "Cached the first %d of %d characters of %r for keyword search.",
            CACHED_DOCUMENT_MAX_CHARS,
            entry.char_count,
            document.name,
        )
    return entry


async def cache_document_text(document: Document, text: str, ctx=None) -> CachedDocument | None:
    """Cache ``document`` from ``text`` already in hand, without reading its file.

    ``update()`` holds the new text when it re-caches an edited document, and
    reading the file again could only add a way to fail after its chunks
    were written.
    """
    entry = _entry_for(document, text)
    if entry is not None:
        await add_data_points([entry], ctx=ctx, index_edges=False)
    return entry


@task_summary("Cached {n} document(s) for keyword search")
async def cache_documents(documents: list[Document], ctx=None) -> list[CachedDocument]:
    """Store each document's full text so BM25 search answers before chunking.

    Args:
        documents: The typed documents ``classify_documents`` produced.
        ctx: Pipeline runtime context, passed through to ``add_data_points`` for
            provenance stamping.

    Returns:
        The ``CachedDocument`` written for each readable document.
    """
    cached: list[CachedDocument] = []
    for document in documents:
        try:
            text = await read_document_text(document)
        except UnicodeDecodeError as error:
            # Deterministic, unlike an I/O error, which fails the item so the
            # document is cached on the next run instead of stamped as done.
            logger.warning(
                "Not caching %r for keyword search: its stored text is not UTF-8 (%s).",
                document.name,
                error,
            )
            continue
        entry = _entry_for(document, text)
        if entry is not None:
            cached.append(entry)

    if cached:
        await add_data_points(cached, ctx=ctx, index_edges=False)
    return cached
