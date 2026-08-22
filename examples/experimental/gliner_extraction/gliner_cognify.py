"""gliner_cognify: a fully LLM-free cognify replacement powered by GLiNER2.

Swaps every LLM stage of cognee's cognify pipeline:
  - graph extraction  -> GLiNER2 entities + relations (encoder model, local CPU)
  - chunk summaries   -> GLiNER2 extractive summaries (top entity-dense
                         sentences + an entity digest; GLiNER cannot *write*
                         text, so summaries are extractive, not abstractive)

The task list mirrors cognify's default one:
  classify_documents -> extract_chunks_from_documents ->
  extract_graph_from_data (GLiNER hook) -> summarize_text_with_gliner ->
  add_data_points

Only embeddings (add_data_points vector indexing) still call an external
embedding model. No LLM completion is invoked anywhere.
"""

import asyncio
import re
from uuid import uuid5

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines import run_pipeline
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization.models import TextSummary

from gliner_graph_extractor import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_RELATION_TYPES,
    _to_knowledge_graph,
)
from parallel_gliner import pack_texts, unpack_results


def _sentence_ranges(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    ranges = []
    offset = 0
    for sentence in sentences:
        start = text.find(sentence, offset)
        if start == -1:
            start = offset
        end = start + len(sentence)
        ranges.append((start, end))
        offset = end
    return ranges


def gliner_extractive_summary(text: str, result: dict, max_sentences: int = 2) -> str:
    """Extractive summary: the most entity-dense sentences plus an entity digest.

    GLiNER2 is encoder-only, so it selects content rather than generating it.
    `result` is a GLiNER entity extraction result with spans for `text`.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    digest_parts = []
    for entity_type, values in result.get("entities", {}).items():
        names = list(dict.fromkeys(v["text"] for v in values))
        if names:
            digest_parts.append(f"{entity_type}: {', '.join(names)}")
    digest = f" Key entities — {'; '.join(digest_parts)}." if digest_parts else ""

    if len(sentences) <= max_sentences:
        return " ".join(sentences) + digest

    ranges = _sentence_ranges(text, sentences)
    scores = [0] * len(sentences)
    for values in result.get("entities", {}).values():
        for span in values:
            for index, (start, end) in enumerate(ranges):
                if start <= span["start"] < end:
                    scores[index] += 1
                    break

    top_indexes = sorted(sorted(range(len(sentences)), key=lambda i: -scores[i])[:max_sentences])
    return " ".join(sentences[i] for i in top_indexes) + digest


async def summarize_text_with_gliner(
    data_chunks: list,
    extractor=None,
    entity_types: dict | list | None = None,
    batch_size: int = 16,
    **kwargs,
) -> list[TextSummary]:
    """Drop-in replacement for cognee's summarize_text — zero LLM calls."""
    if not data_chunks:
        return data_chunks

    # One batched GLiNER call for all chunks in this cognify batch.
    entity_results = await asyncio.to_thread(
        extractor.batch_extract_entities,
        [chunk.text for chunk in data_chunks],
        entity_types or DEFAULT_ENTITY_TYPES,
        batch_size=batch_size,
        include_spans=True,
    )
    summary_texts = [
        gliner_extractive_summary(chunk.text, result)
        for chunk, result in zip(data_chunks, entity_results)
    ]

    return [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            source_chunk_id=str(chunk.id),
            belongs_to_set=chunk.belongs_to_set,
            text=summary_text,
            importance_weight=chunk.importance_weight,
        )
        for chunk, summary_text in zip(data_chunks, summary_texts)
    ]


async def gliner_extract_and_summarize(
    data_chunks: list,
    extractor=None,
    entity_types: dict | list | None = None,
    relation_types: dict | list | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    schema_tuner=None,
    worker_pool=None,
    pack_target_chars: int = 1800,
    ctx=None,
    **kwargs,
) -> list[TextSummary]:
    """Graph extraction AND summaries from ONE batched GLiNER forward pass.

    A single `batch_extract` call (entities with spans + relations) feeds both
    the knowledge-graph construction and the extractive summaries, halving
    GLiNER compute versus running the two tasks separately.

    Chunk texts are PACKED to ~`pack_target_chars` per GLiNER input (spans
    remapped back to source chunks) so forward passes carry content instead
    of padding. Pass a `GLiNERWorkerPool` as `worker_pool` to shard the
    packed inputs across processes; `extractor` is then not needed.

    Pass an `AdaptiveSchemaTuner` as `schema_tuner` to monitor per-batch
    entity density and expand the label set (one discovery LLM call) when a
    batch's coverage drops; the batch is then re-extracted once with the
    expanded schema.
    """
    if not data_chunks:
        return data_chunks

    if schema_tuner is not None:
        entity_types = schema_tuner.entity_types
        relation_types = schema_tuner.relation_types

    texts = [chunk.text for chunk in data_chunks]

    async def _extract(ents, rels):
        packed_texts, packs = pack_texts(texts, target_chars=pack_target_chars)
        if worker_pool is not None:
            packed_results = await worker_pool.extract(
                packed_texts, ents, rels, threshold=threshold, batch_size=batch_size
            )
        else:
            schema = extractor.create_schema().entities(ents).relations(rels)
            packed_results = await asyncio.to_thread(
                extractor.batch_extract,
                packed_texts,
                schema,
                batch_size=batch_size,
                threshold=threshold,
                include_spans=True,
            )
        return unpack_results(packed_results, packs, len(texts))

    results = await _extract(
        entity_types or DEFAULT_ENTITY_TYPES, relation_types or DEFAULT_RELATION_TYPES
    )

    if schema_tuner is not None and await schema_tuner.observe_and_maybe_expand(texts, results):
        # Schema grew — re-extract this batch once with the expanded labels.
        results = await _extract(schema_tuner.entity_types, schema_tuner.relation_types)

    chunk_graphs = [_to_knowledge_graph(result) for result in results]
    await extract_graph_from_data(
        data_chunks,
        KnowledgeGraph,
        ctx=ctx,
        calculate_chunk_graphs=lambda *args, **kw: chunk_graphs,
    )

    return [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            source_chunk_id=str(chunk.id),
            belongs_to_set=chunk.belongs_to_set,
            text=gliner_extractive_summary(chunk.text, result),
            importance_weight=chunk.importance_weight,
        )
        for chunk, result in zip(data_chunks, results)
    ]


class _BackgroundStorage:
    """Overlap storage with extraction: add_data_points runs as background
    asyncio tasks (bounded to `depth` outstanding batches) so the runner
    returns to GLiNER extraction of the next batch while previous batches
    embed and write. depth=2 additionally overlaps one batch's embedding
    network wait with another batch's graph write. `flush()` must be awaited
    after the pipeline to persist the tail and surface any storage error."""

    def __init__(self, depth: int = 1):
        self.depth = depth
        self._pending: list = []

    async def store_data_points(self, data_points: list, ctx=None) -> list:
        while len(self._pending) >= self.depth:
            oldest = self._pending.pop(0)
            await oldest
        self._pending.append(asyncio.create_task(add_data_points(data_points, ctx=ctx)))
        return []

    async def flush(self):
        pending, self._pending = self._pending, []
        for task in pending:
            await task


async def gliner_cognify(
    datasets,
    extractor=None,
    entity_types: dict | list | None = None,
    relation_types: dict | list | None = None,
    chunk_size: int = 1024,
    chunks_per_batch: int = 100,
    gliner_batch_size: int = 32,
    schema_tuner=None,
    workers: int = 0,
    pack_target_chars: int = 1800,
    model_name: str = "fastino/gliner2-base-v1",
    storage_depth: int = 1,
):
    """Run the whole cognify pipeline with GLiNER2 instead of an LLM.

    With `workers > 0`, extraction runs in that many GLiNER worker processes
    (~2 GB RAM each) and `extractor` is not needed. Storage always overlaps
    extraction of the next batch; `storage_depth=2` also overlaps two storage
    batches with each other (embedding wait vs graph write).
    """
    worker_pool = None
    if workers > 0:
        from parallel_gliner import GLiNERWorkerPool

        worker_pool = GLiNERWorkerPool(model_name=model_name, workers=workers)
    elif extractor is None:
        raise ValueError("Provide `extractor` or set `workers` > 0")

    storage = _BackgroundStorage(depth=storage_depth)
    tasks = [
        Task(classify_documents),
        Task(extract_chunks_from_documents, max_chunk_size=chunk_size, chunker=TextChunker),
        Task(
            gliner_extract_and_summarize,
            extractor=extractor,
            entity_types=entity_types,
            relation_types=relation_types,
            batch_size=gliner_batch_size,
            schema_tuner=schema_tuner,
            worker_pool=worker_pool,
            pack_target_chars=pack_target_chars,
            task_config={"batch_size": chunks_per_batch},
        ),
        Task(storage.store_data_points, task_config={"batch_size": chunks_per_batch}),
    ]

    try:
        run_infos = []
        async for run_info in run_pipeline(
            tasks=tasks,
            datasets=datasets,
            pipeline_name="gliner_cognify_pipeline",
            incremental_loading=False,
            # The default connection test pings the LLM; this pipeline never
            # calls one, so skip it (same as cognee's LLM-free code pipeline).
            skip_connection_test=True,
        ):
            run_infos.append(run_info)
        await storage.flush()
        return run_infos
    finally:
        await storage.flush()
        if worker_pool is not None:
            worker_pool.shutdown()
