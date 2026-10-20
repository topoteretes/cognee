"""Memify pipeline: a repository's docstrings as knowledge-graph chunks linked to its code graph.

SearchType.CODE answers structural questions over the enola code graph, but
the graph holds no prose beyond the first paragraph of a symbol's ``doc``
prop, and no completion or chunk retriever reads code nodes. This pipeline
reads the docstrings of every repository whose code graph is in the dataset
(``extract_code_docs``: the whole comments enola wrote to ``docs.jsonl`` in the
snapshot the repository's graph was built from, ``<CodeRepository.path>/.enola``)
and stores them as plain ``DocumentChunk`` nodes -- one per documented symbol
or module, text headed ``file:line`` and ``name (kind)`` -- under one
``CodeDocument`` per file (per package directory for a module doc). A
repository whose snapshot is gone, or no longer the one its graph holds, keeps
its stored docs untouched. Plain chunks on purpose: every existing
retriever (recall, GRAPH_COMPLETION, HYBRID, CHUNKS) already searches
``DocumentChunk_text``, so "what does memify do" is answered from the docstring
text with no retrieval change. ``add_code_doc_edges`` then links each chunk to
its ``CodeSymbol`` with a ``describes`` edge, so a hit reaches the symbol's
file, line, callers and flows in one hop, and sweeps the chunks a re-run no
longer yields, from the graph and from their vector collections.

Ids are deterministic (document: ``code_document_id(repo, file)``; chunk: the
content hash of its text), so a re-run over unchanged docs yields the same
ids: ``store_code_docs`` writes only the points the graph does not hold yet,
no edge is added, a changed docstring replaces exactly its chunk and the
sweep removes nothing else. An unchanged re-run writes and embeds nothing.

Embed-only by default (no LLM call: ``needs_llm=False`` on every task).
``extract_entities=True`` adds the standard cognify extraction and summaries
over the chunks the graph has not summarized yet, one cognify batch at a time.

The code graph routes run the same chain for one repository through
``cognee.tasks.code_docs.code_docs_tail``.
"""

from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.memify import run_memify_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.code_docs.add_code_doc_edges import add_code_doc_edges
from cognee.tasks.code_docs.cognify_code_doc_chunks import (
    cognify_chunks_per_batch,
    cognify_code_doc_chunks,
)
from cognee.tasks.code_docs.extract_code_docs import extract_code_docs
from cognee.tasks.code_docs.store_code_docs import store_code_docs

__all__ = [
    "cognify_chunks_per_batch",
    "cognify_code_doc_chunks",
    "cognify_code_docs_pipeline",
    "get_code_docs_memify_tasks",
]


def get_code_docs_memify_tasks(
    extract_entities: bool = False,
    repo: str | None = None,
    include_modules: bool = True,
) -> tuple[list[Task], list[Task]]:
    """Build the (extraction, enrichment) task lists of the code docs pipeline.

    ``repo`` limits the run to one enola repository label (None: every
    CodeRepository in the dataset) and scopes the sweep the same way;
    ``include_modules`` adds a chunk per module (package) doc. The enrichment
    chain is store_code_docs (the points the graph lacks, through
    add_data_points at cognify's batch size), then add_code_doc_edges, with,
    for ``extract_entities``, cognify_code_doc_chunks in front of it:
    cognify's extract_graph_and_summarize (the generic KnowledgeGraph,
    cognify's defaults) over the not yet summarized chunks only, one cognify
    batch at a time.

    Every task is a coroutine, so the pipeline hands each one the whole run's
    output in one call (only a generator's output is cut into ``batch_size``
    pieces): add_code_doc_edges sees the complete run and its sweep compares
    the stored docs against all of it. The ``batch_size`` on store_code_docs
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
        Task(store_code_docs, task_config={"batch_size": batch_size}, needs_llm=False),
        Task(add_code_doc_edges, repo=repo, needs_llm=False),
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
