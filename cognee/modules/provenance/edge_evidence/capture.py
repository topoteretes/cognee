"""Hot-path facade: capture in memory and flush only at a bounded threshold."""

from typing import Any, Iterable

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk

from .buffer import ProvenanceBuffer
from .config import get_provenance_config
from .persistence import flush_context_provenance


def collect_document_chunks(data_points: Iterable[Any]) -> list[DocumentChunk]:
    """Find every DocumentChunk instance reachable from a storage batch.

    The batch is walked as an object graph rather than through
    ``get_graph_from_model``: that expansion rebuilds nodes as stripped copies,
    which drops both the ``DocumentChunk`` type and the ``_provenance_edges``
    private attribute the buffer reads. Chunks are frequently nested — the
    default cognify batch is ``TextSummary`` objects holding their chunk under
    ``made_from`` — so a top-level ``isinstance`` scan is not enough.
    """
    chunks: list[DocumentChunk] = []
    seen: set[int] = set()
    stack = list(data_points)
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple, set)):
            stack.extend(item)
            continue
        if not isinstance(item, DataPoint) or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, DocumentChunk):
            chunks.append(item)
        stack.extend(
            value
            for field_name in type(item).model_fields
            if (value := getattr(item, field_name, None)) is not None
            and isinstance(value, (DataPoint, list, tuple, set))
        )
    return chunks


async def capture_graph_provenance(
    data_points: Iterable[Any], graph_edges: Iterable[Any], ctx: Any
) -> int:
    """Record chunk→edge support links for the DocumentChunks in a storage batch.

    ``data_points`` is the task's input batch (the original objects, nested
    chunks included); ``graph_edges`` are the expanded edges about to be written.

    Coverage is deliberately narrow: evidence exists only for edges produced
    while storing document chunks (the cognify extraction path). Batches with
    no reachable ``DocumentChunk`` — contradiction edges, ``improve()``
    enrichment, the code-graph route, session bridging — record nothing.
    ``evidence_kind`` on the row is the extension point for those producers.
    """
    config = get_provenance_config()
    if not config.edge_evidence_enabled or ctx is None:
        return 0

    chunks = collect_document_chunks(data_points)
    if not chunks:
        return 0

    buffer = getattr(ctx, "provenance_buffer", None)
    if not isinstance(buffer, ProvenanceBuffer):
        buffer = ProvenanceBuffer()
        ctx.provenance_buffer = buffer

    captured = buffer.capture(chunks=chunks, graph_edges=graph_edges, ctx=ctx)
    if buffer.pending_record_count() >= config.edge_evidence_flush_threshold:
        await flush_context_provenance(ctx)
    return captured
