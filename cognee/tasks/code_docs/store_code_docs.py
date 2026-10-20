"""Store the code doc data points the graph does not hold yet; pass the whole batch on.

add_data_points merges and re-embeds every node it is handed, so a run over an
unchanged repository used to cost the writes and embeddings of every chunk
again. This task takes its place in the code docs chain: it looks the batch's
CodeDocument and DocumentChunk ids up in the graph (an id lookup, not a graph
read), writes the points that are new or changed -- with the documents of
those chunks and every summary or entity extraction it is handed -- and
RETURNS THE WHOLE BATCH, so add_code_doc_edges still sees the complete run for
its edge diff and its sweep.

A chunk's id is derived from its text (see extract_code_docs), so a stored id
means an unchanged docstring. A document's id is (repo, file), so its stored
node is compared on what the run derives for it (symbol count, node sets)
before it is left alone.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.extract_code_docs import _as_int, node_properties
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.code_graph.extract_code_graph import _SWEEP_CHUNK_SIZE
from cognee.tasks.storage.add_data_points import add_data_points

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")


@dataclass
class StoreCounts:
    """What one store pass wrote and what it left alone."""

    # CodeDocuments written: new, changed, or holding a written chunk.
    documents: int = 0
    # DocumentChunks written: a new or changed id, changed tags, or extraction
    # output to store (``contains``, or a summary made from it).
    chunks: int = 0
    # Points that are neither documents nor chunks (TextSummaries): always written.
    extractions: int = 0
    # Stored documents and chunks the batch repeats unchanged.
    skipped: int = 0

    @property
    def written(self) -> int:
        return self.documents + self.chunks + self.extractions


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _document_changed(document: CodeDocument, stored: dict[str, Any]) -> bool:
    """The stored CodeDocument differs in what the run derives for it: symbol count, node sets."""
    if _as_int(stored.get("symbol_count")) != document.symbol_count:
        return True
    return _text(stored.get("source_node_set")) != _text(document.source_node_set)


def _chunk_changed(chunk: DocumentChunk, stored: dict[str, Any]) -> bool:
    """A stored chunk has this chunk's text (the id is its hash); its tags or its place can differ.

    The place is ``chunk_index``: a symbol above this one gained or lost a
    docstring without moving it (a comment line turned into a docstring, say),
    so the text and the id stayed while the position in the document moved. A
    stored row without the field (an adapter answering the core columns only)
    is not called changed on it. The symbol's ``end_line`` in the metadata is
    not compared: a body that grows moves it on every edit, and rewriting the
    chunk for it would re-embed an unchanged docstring each time, so it keeps
    the value from when the docstring was last stored.
    """
    if _text(stored.get("source_node_set")) != _text(chunk.source_node_set):
        return True
    stored_index = _as_int(stored.get("chunk_index"))
    return stored_index is not None and stored_index != chunk.chunk_index


def _batch_chunk(point: DataPoint) -> DocumentChunk | None:
    """The chunk a batch point is, or carries (a TextSummary's ``made_from``)."""
    chunk = point if isinstance(point, DocumentChunk) else getattr(point, "made_from", None)
    return chunk if isinstance(chunk, DocumentChunk) else None


def lookup_ids(data_points: list[DataPoint]) -> list[str]:
    """The ids whose presence decides what is written: the batch's documents and chunks."""
    ids: dict[str, None] = {}
    for point in data_points:
        if isinstance(point, CodeDocument):
            ids[str(point.id)] = None
        chunk = _batch_chunk(point)
        if chunk is not None:
            ids[str(chunk.id)] = None
    return list(ids)


def select_new_code_docs(
    data_points: list[DataPoint], stored: dict[str, dict[str, Any]]
) -> tuple[list[DataPoint], StoreCounts]:
    """The points to write given the stored nodes by id; documents first, then batch order.

    A DocumentChunk is written when its id is not stored, when its stored tags
    differ, or when it carries extraction output (``contains``). Any other
    point -- a TextSummary made from a chunk -- is always written, its chunk
    with it. A CodeDocument is written when it is new or changed, or when one
    of its chunks is written (the chunk's ``is_part_of`` would write it
    anyway; listing it keeps the write explicit).
    """
    counts = StoreCounts()
    documents: list[CodeDocument] = []
    written_document_ids: set[str] = set()
    to_write: list[DataPoint] = []
    for point in data_points:
        if isinstance(point, CodeDocument):
            documents.append(point)
            continue
        chunk = _batch_chunk(point)
        if isinstance(point, DocumentChunk):
            stored_chunk = stored.get(str(point.id))
            if (
                stored_chunk is not None
                and not point.contains
                and not _chunk_changed(point, stored_chunk)
            ):
                counts.skipped += 1
                continue
            counts.chunks += 1
        else:
            counts.extractions += 1
            if chunk is not None and str(chunk.id) not in stored:
                counts.chunks += 1
        to_write.append(point)
        document = getattr(chunk, "is_part_of", None)
        if isinstance(document, CodeDocument):
            written_document_ids.add(str(document.id))

    document_points: list[DataPoint] = []
    for document in documents:
        stored_document = stored.get(str(document.id))
        if (
            str(document.id) in written_document_ids
            or stored_document is None
            or _document_changed(document, stored_document)
        ):
            written_document_ids.add(str(document.id))
            document_points.append(document)
        else:
            counts.skipped += 1
    counts.documents = len(written_document_ids)
    return document_points + to_write, counts


async def _stored_nodes(graph_engine, node_ids: list[str]) -> dict[str, dict[str, Any]]:
    """The stored nodes among ``node_ids``, by id: an id lookup, no graph read."""
    stored: dict[str, dict[str, Any]] = {}
    for start in range(0, len(node_ids), _SWEEP_CHUNK_SIZE):
        rows = await graph_engine.get_nodes(node_ids[start : start + _SWEEP_CHUNK_SIZE])
        for row in rows or ():
            properties = node_properties(row)
            if properties.get("id") is not None:
                stored[str(properties["id"])] = properties
    return stored


async def store_new_code_docs(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = False,
) -> StoreCounts:
    """Write the batch's new or changed code docs, and every extraction, through add_data_points.

    The counts are what store_code_docs logs and code_docs_tail reports. An
    unchanged re-run writes nothing and never calls add_data_points.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if not data_points:
        return StoreCounts()
    ids = lookup_ids(data_points)
    stored = await _stored_nodes(await get_graph_engine(), ids) if ids else {}
    to_write, counts = select_new_code_docs(data_points, stored)
    if to_write:
        await add_data_points(to_write, ctx=ctx, graph_only=graph_only)
    logger.info(
        "Code docs store: %d document(s), %d chunk(s) and %d extraction(s) written; "
        "%d stored point(s) unchanged.",
        counts.documents,
        counts.chunks,
        counts.extractions,
        counts.skipped,
    )
    return counts


@task_summary("Stored or kept {n} code doc data point(s)")
async def store_code_docs(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = False,
) -> list[DataPoint]:
    """Store the batch's new code docs; pass the whole batch on.

    Takes add_data_points' place in the code docs chain (after extract_code_docs,
    or cognify_code_doc_chunks in entity mode; before add_code_doc_edges) and
    receives the whole run: CodeDocuments and DocumentChunks, or TextSummaries
    made from the chunks. Only the points the graph does not hold yet are
    written (see store_new_code_docs); ``ctx`` attributes them to the context's
    data item exactly as add_data_points would; ``graph_only`` stores without a
    vector engine. Passthrough: returns data_points unchanged, so the link task
    sees the complete run.
    """
    await store_new_code_docs(data_points, ctx=ctx, graph_only=graph_only)
    return data_points
