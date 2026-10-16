"""Cognify's entity extraction and summaries over the docstring chunks, new chunks only.

The optional LLM stage of the code docs chain (``extract_entities``). It runs
cognify's extract_graph_and_summarize over the run's DocumentChunks one
cognify batch at a time, and only over the chunks the graph has not
summarized yet: a chunk id is its text's hash, so a stored chunk that a
TextSummary was ``made_from`` went through extraction in an earlier run, and
its entities and summary are in the graph already. Those chunks pass through
untouched, so the whole run still reaches storage and the link task's sweep.
"""

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.summarization.models import TextSummary

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")

MADE_FROM_RELATIONSHIP = "made_from"


def cognify_chunks_per_batch() -> int:
    """The batch size cognify stores chunks with: its config, 2000 when unset."""
    chunks_per_batch = get_cognify_config().chunks_per_batch
    return chunks_per_batch if chunks_per_batch is not None else 2000


async def summarized_chunk_ids(graph_engine, chunk_ids: set[str]) -> set[str]:
    """Those of ``chunk_ids`` that a stored TextSummary was ``made_from``.

    One filtered read of the summaries and chunks (a filtered read returns an
    edge only when both endpoints match the filter, so both types are in).
    """
    if not chunk_ids:
        return set()
    _nodes, edges = await graph_engine.get_filtered_graph_data(
        [{"type": [TextSummary.__name__, DocumentChunk.__name__]}]
    )
    return {
        str(edge[1])
        for edge in edges or ()
        if len(edge) >= 3 and str(edge[2]) == MADE_FROM_RELATIONSHIP and str(edge[1]) in chunk_ids
    }


async def cognify_code_doc_chunks(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_model: type[BaseModel] = KnowledgeGraph,
    chunks_per_batch: int | None = None,
) -> list[DataPoint]:
    """Run cognify's extract_graph_and_summarize over the run's new chunks, a cognify batch at a time.

    The whole run arrives in one call (every task of the chain is a coroutine,
    see get_code_docs_memify_tasks), and extract_graph_and_summarize fires one
    LLM extraction and one summary request per chunk it is handed, all at
    once. So the chunks are cut into ``chunks_per_batch`` pieces (cognify's
    chunks_per_batch when None) and each piece is awaited before the next:
    the LLM sees at most one cognify batch of concurrent requests, as under
    cognify's own streaming chunker, while the next task still receives the
    whole run.

    Only chunks without a stored summary are extracted (summarized_chunk_ids);
    the others are returned as they are, after the extraction results. Only
    the chunks go in (extraction reads ``.text``); the documents are not lost,
    since each chunk references its document through ``is_part_of`` and the
    store writes the whole object graph, exactly how the cognify chain stores
    documents. A batch without chunks is returned unchanged, so an empty run
    still reaches the sweep. Returns what extraction returned for each piece
    -- summaries, or chunks as themselves -- then the unchanged chunks.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    chunks = [point for point in data_points if isinstance(point, DocumentChunk)]
    if not chunks:
        return data_points
    if not chunks_per_batch or chunks_per_batch <= 0:
        chunks_per_batch = cognify_chunks_per_batch()

    graph_engine = await get_graph_engine()
    summarized = await summarized_chunk_ids(graph_engine, {str(chunk.id) for chunk in chunks})
    pending = [chunk for chunk in chunks if str(chunk.id) not in summarized]
    unchanged = [chunk for chunk in chunks if str(chunk.id) in summarized]
    logger.info(
        "Code docs entity extraction: %d chunk(s) to extract, %d already summarized.",
        len(pending),
        len(unchanged),
    )

    results: list[DataPoint] = []
    for start in range(0, len(pending), chunks_per_batch):
        results.extend(
            await extract_graph_and_summarize(
                pending[start : start + chunks_per_batch], graph_model=graph_model, ctx=ctx
            )
        )
    return results + unchanged
