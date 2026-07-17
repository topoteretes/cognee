"""Hot-path facade: capture in memory and flush only at a bounded threshold."""

from typing import Any, Iterable

from cognee.modules.chunking.models import DocumentChunk

from .buffer import ProvenanceBuffer
from .config import get_provenance_config
from .persistence import flush_context_provenance


async def capture_graph_provenance(
    data_points: Iterable[Any], graph_edges: Iterable[Any], ctx: Any
) -> int:
    config = get_provenance_config()
    if not config.provenance_enabled or ctx is None:
        return 0

    chunks = [item for item in data_points if isinstance(item, DocumentChunk)]
    if not chunks:
        return 0

    buffer = getattr(ctx, "provenance_buffer", None)
    if not isinstance(buffer, ProvenanceBuffer):
        buffer = ProvenanceBuffer()
        ctx.provenance_buffer = buffer

    captured = buffer.capture(chunks=chunks, graph_edges=graph_edges, ctx=ctx)
    if buffer.pending_record_count() >= config.provenance_flush_threshold:
        await flush_context_provenance(ctx)
    return captured
