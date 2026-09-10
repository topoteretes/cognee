"""LLM-free cognify tasks backed by GLiNER.

``get_gliner_tasks`` returns the complete task list
``classify_documents -> prepare_gliner_schema -> extract_chunks_from_documents ->
extract_graph_and_summarize_with_gliner -> add_data_points``. Run it with
``cognee.run_custom_pipeline(tasks=..., dataset=..., pipeline_name="cognify_pipeline")``;
the pipeline loads the dataset's Data records itself, exactly as ``cognify()`` does.

The default ``cognify()`` pipeline keeps its LLM task list unless told
otherwise: ``cognify(extractor="gliner")`` (or the
``GRAPH_EXTRACTOR=gliner`` setting) selects this list instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.pipelines.tasks.task import Task, task_summary
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import detect_contradictions
from cognee.tasks.graph.exceptions import InvalidDataChunksError
from cognee.tasks.graph.extract_graph_from_data import extract_graph_from_data
from cognee.tasks.graph.resolve_temporal_contradictions import resolve_temporal_contradictions
from cognee.tasks.provenance import record_provenance
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
from .schema import (
    MAX_SKETCH_WORDS,
    GlinerSchema,
    LabelSpec,
    make_document_sketch,
    resolve_schema,
    schema_from_label_bank,
)
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


@task_summary("Prepared GLiNER schema for {n} document(s)")
async def prepare_gliner_schema(
    documents: list[Document],
    schema: GlinerSchema,
    max_chunk_size: int,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    chunker=TextChunker,
) -> list[Document]:
    """Attach one closed schema to each document before it is chunked."""
    extractor = None
    for document in documents:
        document_schema = schema
        if document_schema.is_empty:
            if extractor is None:
                extractor = await get_extractor(model_name)
            text_parts = [
                chunk.text
                async for chunk in document.read(
                    max_chunk_size=max_chunk_size,
                    chunker_cls=chunker,
                )
            ]
            model_max_words = getattr(getattr(extractor, "config", None), "max_len", None)
            sketch_max_words = min(MAX_SKETCH_WORDS, model_max_words or MAX_SKETCH_WORDS)
            sketch = make_document_sketch(
                "".join(text_parts),
                max_words=sketch_max_words,
            )
            document_schema = await asyncio.to_thread(
                schema_from_label_bank,
                extractor,
                sketch,
                threshold=threshold,
            )

        document._gliner_schema = document_schema
        logger.info(
            "GLiNER schema for %s resolved from %s: %d entity types, %d relation types",
            document.name,
            document_schema.source,
            len(document_schema.entity_types),
            len(document_schema.relation_types),
        )

    return documents


@task_summary("GLiNER-extracted graph and summaries for {n} chunk(s)")
async def extract_graph_and_summarize_with_gliner(
    data_chunks: list[DocumentChunk],
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

    schema = data_chunks[0].is_part_of._gliner_schema
    if schema is None:
        raise RuntimeError("GLiNER schema was not prepared for this document")

    texts = [chunk.text for chunk in data_chunks]
    if schema.is_empty:
        results = [{} for _ in texts]
    else:
        extractor = await get_extractor(options.model_name)
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
    *,
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

    This is the task ``get_gliner_tasks`` places fourth and the one
    ``get_gliner_tasks`` places in its extraction step. Raises
    :class:`GlinerNotInstalledError` when ``gliner2`` is not installed and
    ``ValueError`` on bad options.
    """
    require_gliner2()
    options = GlinerOptions(
        model_name=model_name,
        threshold=threshold,
        batch_size=gliner_batch_size,
        window_words=window_words,
        window_overlap_words=window_overlap_words,
    )
    return Task(
        extract_graph_and_summarize_with_gliner,
        stats=stats if stats is not None else GlinerRunStats(),
        options=options,
        config=config,
        chunk_attachment=chunk_attachment,
        task_config={"batch_size": chunks_per_batch},
        needs_llm=False,  # local model; a pipeline of needs_llm=False tasks skips the LLM probe
    )


def build_gliner_schema_task(
    entity_types: LabelSpec | None = None,
    relation_types: LabelSpec | None = None,
    *,
    ontology_file_path: str | None = None,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    max_chunk_size: int,
    chunker=TextChunker,
) -> Task:
    """Build the document-level schema preparation task."""
    require_gliner2()
    schema = resolve_schema(
        entity_types,
        relation_types,
        ontology_file_path=ontology_file_path,
    )
    return Task(
        prepare_gliner_schema,
        schema=schema,
        max_chunk_size=max_chunk_size,
        model_name=model_name,
        threshold=threshold,
        chunker=chunker,
        needs_llm=False,
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
    track_provenance: bool = False,
    check_contradictions: bool = False,
    functional_relationships: Collection[str] | None = None,
    stats: GlinerRunStats | None = None,
) -> list[Task]:
    """Build the GLiNER cognify task list.

    ``entity_types`` / ``relation_types`` accept a list of names or a
    ``name -> description`` mapping and, when given, are the whole schema (no
    ontology read, no bank probe). Otherwise the schema falls through to the
    configured OWL ontology (``ontology_file_path`` overrides
    ``ONTOLOGY_FILE_PATH``), then to the frozen label banks probed once per
    document sketch. ``stats`` is filled in as the run progresses.

    Embeddings in ``add_data_points`` still run; graph extraction and summaries
    are LLM-free. Optional contradiction detection still uses the LLM. Raises
    :class:`GlinerNotInstalledError` when ``gliner2`` is not installed.
    """
    if chunks_per_batch is None:
        configured = get_cognify_config().chunks_per_batch
        chunks_per_batch = configured if configured is not None else 2000

    max_chunk_size = chunk_size or await get_max_chunk_tokens()
    schema_task = build_gliner_schema_task(
        entity_types,
        relation_types,
        ontology_file_path=ontology_file_path,
        model_name=model_name,
        threshold=threshold,
        max_chunk_size=max_chunk_size,
        chunker=chunker,
    )
    extraction_task = build_gliner_extraction_task(
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

    tasks = [
        Task(classify_documents, needs_llm=False),
        schema_task,
        Task(
            extract_chunks_from_documents,
            max_chunk_size=max_chunk_size,
            chunker=chunker,
            needs_llm=False,
        ),
        extraction_task,
        Task(
            add_data_points,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
            needs_llm=False,
        ),
    ]

    if track_provenance:
        tasks.append(
            Task(record_provenance, task_config={"batch_size": chunks_per_batch}, needs_llm=False)
        )

    if check_contradictions:
        tasks.append(Task(detect_contradictions, task_config={"batch_size": chunks_per_batch}))

    if functional_relationships:
        tasks.append(
            Task(
                resolve_temporal_contradictions,
                functional_relationships=functional_relationships,
                task_config={"batch_size": chunks_per_batch},
                needs_llm=False,
            )
        )

    return tasks
