"""The code docs chain as one call: extract, (cognify,) store, link and sweep.

The code graph routes (cognify's CODE_REPO and CODE routes, and the
``remember(content_type="code")`` task list) run it after their code graph
tasks for the repository they just indexed; the memify pipeline
(``cognee/memify_pipelines/cognify_code_docs.py``) runs the same functions as
its task list. Sharing them is what keeps the two paths in agreement.
"""

import os
from typing import TYPE_CHECKING, Optional

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_docs.add_code_doc_edges import link_code_docs
from cognee.tasks.code_docs.cognify_code_doc_chunks import cognify_code_doc_chunks
from cognee.tasks.code_docs.extract_code_docs import extract_code_docs
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.code_docs.store_code_docs import store_new_code_docs

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_docs")


async def code_docs_tail(
    repo: str | None,
    ctx: Optional["PipelineContext"] = None,
    *,
    extract_entities: bool = False,
    graph_only: bool = False,
    snapshot_dir: str | os.PathLike | None = None,
) -> dict[str, int]:
    """Document ``repo`` (None: every repository in the dataset) and return the counts.

    extract_code_docs -> [cognify_code_doc_chunks when ``extract_entities``] ->
    store_code_docs -> add_code_doc_edges, with ``ctx`` on every step: the
    stored points are attributed to the context's data item (the repository
    manifest or the code file on the routes), so ``forget`` of that item
    removes them with the code graph, and the sweep retires their chunks
    through the provenance planner. ``repo`` scopes both the extraction and
    the sweep. Embed-only unless ``extract_entities``; ``graph_only`` stores
    the graph nodes without a vector engine. ``snapshot_dir`` is the enola
    snapshot the caller just loaded: the routes pass it, so the docs are read
    from its ``docs.jsonl`` and not from ``<CodeRepository.path>/.enola``
    (which, once an unchanged snapshot is skipped, can name an earlier run's
    staging directory), see extract_code_docs. The link task's sweep follows
    the extraction's record of what it read (CodeDocsBatch), handed on with
    the batch.

    Returns ``{"documents", "chunks", "new", "edges", "swept"}``: the
    CodeDocuments and DocumentChunks the run derived, the chunks written to
    storage (new or changed docstrings; an unchanged re-run writes none), the
    describes edges written, and the stale nodes swept (documents, chunks,
    summaries, entities, and what the planner retired with an owned chunk).
    """
    data_points = await extract_code_docs(repo=repo, ctx=ctx, snapshot_dir=snapshot_dir)
    documents = sum(1 for point in data_points if isinstance(point, CodeDocument))
    chunks = sum(1 for point in data_points if isinstance(point, DocumentChunk))
    if extract_entities:
        data_points = await cognify_code_doc_chunks(data_points, ctx=ctx)
    stored = await store_new_code_docs(data_points, ctx=ctx, graph_only=graph_only)
    linked = await link_code_docs(data_points, ctx=ctx, repo=repo)
    counts = {
        "documents": documents,
        "chunks": chunks,
        "new": stored.chunks,
        "edges": linked.edges,
        "swept": linked.swept.total,
    }
    logger.info(
        "Code docs: %d document(s) and %d chunk(s) derived; %d chunk(s) stored, "
        "%d describes edge(s) written, %d stale node(s) swept.",
        counts["documents"],
        counts["chunks"],
        counts["new"],
        counts["edges"],
        counts["swept"],
    )
    return counts
