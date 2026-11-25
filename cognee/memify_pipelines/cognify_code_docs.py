"""Memify pipeline: a repository's docstrings as knowledge-graph chunks linked to its code graph.

SearchType.CODE answers structural questions over the enola code graph, but
the graph holds no prose beyond the first paragraph of a symbol's ``doc``
prop, and no completion or chunk retriever reads code nodes. This pipeline
reads the docstrings of every repository whose code graph is in the dataset
(``extract_code_docs``: from the source files on the server, enola's ``doc``
prop as fallback) and stores them as plain ``DocumentChunk`` nodes -- one per
documented symbol, text headed ``file:line`` and the signature -- under one
``CodeDocument`` per source file. Plain chunks on purpose: every existing
retriever (recall, GRAPH_COMPLETION, HYBRID, CHUNKS) already searches
``DocumentChunk_text``, so "what does memify do" is answered from the docstring
text with no retrieval change. ``add_code_doc_edges`` then links each chunk to
its ``CodeSymbol`` with a ``describes`` edge, so a hit reaches the symbol's
file, line, callers and flows in one hop, and sweeps the chunks a re-run no
longer yields, from the graph and from their vector collections.

Ids are deterministic (document: ``code_document_id(repo, file)``; chunk: the
content hash of its text), so a re-run over unchanged sources yields the same
ids: no node or edge is added, a changed docstring replaces exactly its chunk
and the sweep removes nothing else. The nodes themselves are stored again on
every run (add_data_points merges and re-embeds what it is handed), so an
unchanged run still costs the writes.

Embed-only by default (no LLM call: ``needs_llm=False`` on every task).
``extract_entities=True`` adds the standard cognify extraction and summaries
over the chunks before they are stored, one cognify batch at a time.
"""

from typing import TYPE_CHECKING, Optional
from uuid import UUID

from pydantic import BaseModel

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.memify import run_memify_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.pipelines.types import Drop
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.code_docs.add_code_doc_edges import add_code_doc_edges
from cognee.tasks.code_docs.extract_code_docs import extract_code_docs
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.storage import add_data_points

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext


def cognify_chunks_per_batch() -> int:
    """The batch size cognify stores chunks with: its config, 2000 when unset."""
    chunks_per_batch = get_cognify_config().chunks_per_batch
    return chunks_per_batch if chunks_per_batch is not None else 2000


async def cognify_code_doc_chunks(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_model: type[BaseModel] = KnowledgeGraph,
    chunks_per_batch: int | None = None,
):
    """Run cognify's extract_graph_and_summarize over the batch's chunks, a cognify batch at a time.

    The whole run arrives in one call (every task of the chain is a coroutine,
    see get_code_docs_memify_tasks), and extract_graph_and_summarize fires one
    LLM extraction and one summary request per chunk it is handed, all at
    once. So the chunks are cut into ``chunks_per_batch`` pieces (cognify's
    chunks_per_batch when None) and each piece is awaited before the next:
    the LLM sees at most one cognify batch of concurrent requests, as under
    cognify's own streaming chunker, while the next task still receives the
    whole run. Only the chunks go in (extraction reads ``.text``); the
    documents are not lost, since each chunk references its document through
    ``is_part_of`` and add_data_points writes the whole object graph, exactly
    how the cognify chain stores documents. A batch without chunks is dropped,
    since extraction rejects an empty list. Returns what extraction returned
    for each piece -- summaries, or chunks as themselves -- in chunk order.
    """
    chunks = [point for point in data_points if isinstance(point, DocumentChunk)]
    if not chunks:
        return Drop
    if not chunks_per_batch or chunks_per_batch <= 0:
        chunks_per_batch = cognify_chunks_per_batch()
    results: list[DataPoint] = []
    for start in range(0, len(chunks), chunks_per_batch):
        results.extend(
            await extract_graph_and_summarize(
                chunks[start : start + chunks_per_batch], graph_model=graph_model, ctx=ctx
            )
        )
    return results


def get_code_docs_memify_tasks(
    extract_entities: bool = False,
    repo: str | None = None,
    include_modules: bool = True,
) -> tuple[list[Task], list[Task]]:
    """Build the (extraction, enrichment) task lists of the code docs pipeline.

    ``repo`` limits the run to one enola repository label (None: every
    CodeRepository in the dataset); ``include_modules`` adds a chunk per
    file-level docstring. The enrichment chain is cognify's storage chain --
    add_data_points at cognify's batch size, then add_code_doc_edges -- with,
    for ``extract_entities``, cognify_code_doc_chunks in front of it: cognify's
    extract_graph_and_summarize (the generic KnowledgeGraph, cognify's
    defaults) over the chunks only, one cognify batch at a time.

    Every task is a coroutine, so the pipeline hands each one the whole run's
    output in one call (only a generator's output is cut into ``batch_size``
    pieces): add_code_doc_edges sees the complete run and its sweep compares
    the stored docs against all of it. The ``batch_size`` on add_data_points
    is cognify's; it would take effect only if a task before it streamed.
    """
    batch_size = cognify_chunks_per_batch()
    extraction_tasks = [
        Task(extract_code_docs, repo=repo, include_modules=include_modules, needs_llm=False),
    ]
    enrichment_tasks = []
    if extract_entities:
        enrichment_tasks.append(
            Task(cognify_code_doc_chunks, graph_model=KnowledgeGraph, chunks_per_batch=batch_size)
        )
    enrichment_tasks += [
        Task(add_data_points, task_config={"batch_size": batch_size}, needs_llm=False),
        Task(add_code_doc_edges, needs_llm=False),
    ]
    return extraction_tasks, enrichment_tasks


async def cognify_code_docs_pipeline(
    user: User,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
    extract_entities: bool = False,
    repo: str | None = None,
    include_modules: bool = True,
):
    """Store the dataset's repositories' docstrings as chunks linked to their code graph.

    Run it once the code graph is in the dataset (``remember(..., content_type="code")``,
    an add() of a repository, or get_code_graph_tasks). Embed-only unless
    ``extract_entities``. ``data=[{}]`` seeds the extraction without projecting the
    dataset's graph, as the other graph-reading memify pipelines do; the task reads
    the code graph itself. Returns the memify runner's result.
    """
    extraction_tasks, enrichment_tasks = get_code_docs_memify_tasks(
        extract_entities=extract_entities, repo=repo, include_modules=include_modules
    )
    return await run_memify_tasks(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )
