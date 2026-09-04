"""LLM-free cognify tasks backed by GLiNER.

``get_gliner_tasks`` returns the four-step task list
``classify_documents -> extract_chunks_from_documents ->
extract_graph_and_summarize_with_gliner -> add_data_points``. Run it with
``cognee.run_custom_pipeline(tasks=..., dataset=..., pipeline_name="cognify_pipeline")``;
the pipeline loads the dataset's Data records itself, exactly as ``cognify()`` does.

The default ``cognify()`` pipeline keeps its LLM extraction unless told
otherwise: ``cognify(graph_extraction_backend="gliner")`` (or the
``GRAPH_EXTRACTION_BACKEND=gliner`` setting) swaps in
``build_gliner_extraction_task`` for ``extract_graph_and_summarize`` and leaves
every other task in place.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.pipelines.tasks.task import Task, task_summary
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph.exceptions import InvalidDataChunksError
from cognee.tasks.graph.extract_graph_from_data import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization.models import TextSummary

from .extractor import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_OVERLAP_WORDS,
    DEFAULT_WINDOW_WORDS,
    extract_batch_async,
    get_extractor,
    require_gliner2,
)
from .mapping import map_gliner_result
from .schema import GlinerSchema, LabelSpec, as_label_map, resolve_schema
from .summary import build_text_summary

logger = get_logger("gliner.tasks")


@dataclass
class GlinerRunStats:
    """Counters the task fills in; pass one to ``get_gliner_tasks(stats=...)`` to read them."""

    chunks: int = 0
    nodes: int = 0
    candidate_edges: int = 0
    kept_edges: int = 0
    schema: GlinerSchema | None = None

    @property
    def dropped_edges(self) -> int:
        return self.candidate_edges - self.kept_edges


@dataclass
class GlinerOptions:
    model_name: str = DEFAULT_MODEL
    threshold: float = DEFAULT_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    window_words: int = DEFAULT_WINDOW_WORDS
    window_overlap_words: int = DEFAULT_WINDOW_OVERLAP_WORDS

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.window_words < 1 or not 0 <= self.window_overlap_words < self.window_words:
            raise ValueError("window_overlap_words must be >= 0 and smaller than window_words")


class SchemaState:
    """Resolves the closed schema on the first batch, then freezes it for the run.

    Chunks do not exist when ``get_gliner_tasks`` returns, so the bank probe can
    only run inside the task. Resolving per batch would make the graph's type
    vocabulary depend on batch composition and order, hence the freeze.
    """

    def __init__(
        self,
        entity_types: LabelSpec | None,
        relation_types: LabelSpec | None,
        *,
        ontology_file_path: str | None,
        options: GlinerOptions,
    ):
        self.entity_types = as_label_map(entity_types)
        self.relation_types = as_label_map(relation_types)
        self.ontology_file_path = ontology_file_path
        self.options = options
        self._resolved: GlinerSchema | None = None
        self._lock = asyncio.Lock()

    @property
    def resolved(self) -> GlinerSchema | None:
        return self._resolved

    async def get(self, extractor, probe_texts: list[str]) -> GlinerSchema:
        if self._resolved is not None:
            return self._resolved
        async with self._lock:
            if self._resolved is None:
                self._resolved = await asyncio.to_thread(
                    resolve_schema,
                    self.entity_types,
                    self.relation_types,
                    extractor=extractor,
                    probe_texts=probe_texts,
                    ontology_file_path=self.ontology_file_path,
                    threshold=self.options.threshold,
                    batch_size=self.options.batch_size,
                    window_words=self.options.window_words,
                    window_overlap_words=self.options.window_overlap_words,
                )
                logger.info(
                    "GLiNER schema resolved from %s: %d entity types, %d relation types",
                    self._resolved.source,
                    len(self._resolved.entity_types),
                    len(self._resolved.relation_types),
                )
            return self._resolved


@task_summary("GLiNER-extracted graph and summaries for {n} chunk(s)")
async def extract_graph_and_summarize_with_gliner(
    data_chunks: list[DocumentChunk],
    schema_state: SchemaState,
    stats: GlinerRunStats,
    options: GlinerOptions,
    config: Config | None = None,
    chunk_attachment: Literal["direct", "all"] | None = None,
    ctx=None,
) -> list[TextSummary]:
    """One batched GLiNER extract per task batch: writes the chunk graphs, returns summaries.

    Replaces ``extract_graph_and_summarize`` on this path. No
    ``extract_content_graph`` and no ``extract_summary`` calls are made; the
    graphs are handed to ``extract_graph_from_data`` through its
    ``calculate_chunk_graphs`` hook so post-extraction ontology matching and
    chunk attachment behave exactly as on the LLM path.
    """
    if not isinstance(data_chunks, list):
        raise InvalidDataChunksError("must be a list of DocumentChunk.")
    if not data_chunks:
        return []
    if not all(hasattr(chunk, "text") for chunk in data_chunks):
        raise InvalidDataChunksError("each chunk must have a 'text' attribute")

    extractor = await get_extractor(options.model_name)
    texts = [chunk.text for chunk in data_chunks]
    schema = await schema_state.get(extractor, texts)

    results = await extract_batch_async(
        extractor,
        texts,
        schema,
        threshold=options.threshold,
        batch_size=options.batch_size,
        window_words=options.window_words,
        window_overlap_words=options.window_overlap_words,
    )
    if len(results) != len(data_chunks):
        raise RuntimeError(f"GLiNER returned {len(results)} results for {len(data_chunks)} chunks")

    mapped = [map_gliner_result(result) for result in results]
    graphs = [item.graph for item in mapped]

    stats.schema = schema
    stats.chunks += len(data_chunks)
    stats.nodes += sum(len(graph.nodes) for graph in graphs)
    stats.candidate_edges += sum(item.candidate_edges for item in mapped)
    stats.kept_edges += sum(item.kept_edges for item in mapped)

    async def precomputed_graphs(*_args, **_kwargs):
        return graphs

    await extract_graph_from_data(
        data_chunks,
        KnowledgeGraph,
        config=config,
        ctx=ctx,
        chunk_attachment=chunk_attachment,
        calculate_chunk_graphs=precomputed_graphs,
    )

    return [build_text_summary(chunk, graph) for chunk, graph in zip(data_chunks, graphs)]


def build_gliner_extraction_task(
    entity_types: LabelSpec | None = None,
    relation_types: LabelSpec | None = None,
    *,
    ontology_file_path: str | None = None,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    gliner_batch_size: int = DEFAULT_BATCH_SIZE,
    window_words: int = DEFAULT_WINDOW_WORDS,
    window_overlap_words: int = DEFAULT_WINDOW_OVERLAP_WORDS,
    chunks_per_batch: int = 2000,
    config: Config | None = None,
    chunk_attachment: Literal["direct", "all"] | None = None,
    stats: GlinerRunStats | None = None,
) -> Task:
    """Build only the GLiNER extract+summarize ``Task``.

    This is the task ``get_gliner_tasks`` places third and the one
    ``cognify(graph_extraction_backend="gliner")`` swaps in for
    ``extract_graph_and_summarize``. Raises :class:`GlinerNotInstalledError`
    when ``gliner2`` is not installed and ``ValueError`` on bad options or more
    caller labels than GLiNER accepts.
    """
    require_gliner2()
    options = GlinerOptions(
        model_name=model_name,
        threshold=threshold,
        batch_size=gliner_batch_size,
        window_words=window_words,
        window_overlap_words=window_overlap_words,
    )
    schema_state = SchemaState(
        entity_types, relation_types, ontology_file_path=ontology_file_path, options=options
    )
    # Validate caller labels now rather than on the first batch.
    if schema_state.entity_types or schema_state.relation_types:
        resolve_schema(schema_state.entity_types, schema_state.relation_types)

    return Task(
        extract_graph_and_summarize_with_gliner,
        schema_state=schema_state,
        stats=stats if stats is not None else GlinerRunStats(),
        options=options,
        config=config,
        chunk_attachment=chunk_attachment,
        task_config={"batch_size": chunks_per_batch},
    )


async def get_gliner_tasks(
    entity_types: LabelSpec | None = None,
    relation_types: LabelSpec | None = None,
    *,
    ontology_file_path: str | None = None,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    gliner_batch_size: int = DEFAULT_BATCH_SIZE,
    window_words: int = DEFAULT_WINDOW_WORDS,
    window_overlap_words: int = DEFAULT_WINDOW_OVERLAP_WORDS,
    chunk_size: int | None = None,
    chunker=TextChunker,
    chunks_per_batch: int | None = None,
    config: Config | None = None,
    chunk_attachment: Literal["direct", "all"] | None = None,
    embed_triplets: bool = False,
    stats: GlinerRunStats | None = None,
) -> list[Task]:
    """Build the LLM-free cognify task list.

    ``entity_types`` / ``relation_types`` accept a list of names or a
    ``name -> description`` mapping and, when given, are the whole schema (no
    ontology read, no bank probe). Otherwise the schema falls through to the
    configured OWL ontology (``ontology_file_path`` overrides
    ``ONTOLOGY_FILE_PATH``), then to the frozen label banks probed on the first
    batch. ``stats`` is filled in as the run progresses.

    Embeddings in ``add_data_points`` still run; only the graph and summaries
    are LLM-free. Raises :class:`GlinerNotInstalledError` when ``gliner2`` is
    not installed.
    """
    if chunks_per_batch is None:
        configured = get_cognify_config().chunks_per_batch
        chunks_per_batch = configured if configured is not None else 2000

    extraction_task = build_gliner_extraction_task(
        entity_types,
        relation_types,
        ontology_file_path=ontology_file_path,
        model_name=model_name,
        threshold=threshold,
        gliner_batch_size=gliner_batch_size,
        window_words=window_words,
        window_overlap_words=window_overlap_words,
        chunks_per_batch=chunks_per_batch,
        config=config,
        chunk_attachment=chunk_attachment,
        stats=stats,
    )

    return [
        Task(classify_documents),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or await get_max_chunk_tokens(),
            chunker=chunker,
        ),
        extraction_task,
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        ),
    ]
