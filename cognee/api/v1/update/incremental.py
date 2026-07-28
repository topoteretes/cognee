"""Chunk-level incremental update: diff, delete affected chunks, re-ingest.

Flow (only runs from the update endpoint):

1. Enforce dataset-level "write" permission and data membership/ownership
   (same checks as the raw-document read endpoint).
2. Read the OLD processed text from ``Data.raw_data_location`` and the stored
   chunk nodes from the graph, ordered by their position in that text.
3. Run ``add()`` so the new file goes through the normal add pipeline (loaders
   store the new processed text and update the ``Data`` row).
4. Diff old vs new text; expand the edit to affected chunk boundaries; split
   the replacement region into balanced chunks under ``max_chunk_size``.
5. Delete the affected chunks (+ summaries + chunk-orphaned entities) from the
   graph and vector stores.
6. Re-ingest ONLY the new chunks through the standard graph-extraction and
   storage tasks, attributed to the same ``data_id`` via ``PipelineContext``.

Raises IncrementalUpdateNotPossible when preconditions fail (first ingestion,
non-text data, stored chunks not tiling the stored text) — the caller decides
to run the full update instead.
"""

import json
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import select

from cognee.api.v1.add import add
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.locks import dataset_lock
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlan,
    IncrementalPlanError,
    compute_incremental_plan,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_dataset, get_data
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.graph.methods.delete_chunks_incremental import delete_chunks_incremental
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_data_item import DataItemStatus
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.extract_graph_from_data import extract_graph_from_data
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tasks.summarization.summarize_text import summarize_text

logger = get_logger("incremental_update")

PIPELINE_NAME = "cognify_pipeline"  # attribute to the cognify pipeline's status key


class IncrementalUpdateNotPossible(Exception):
    """Preconditions for a chunk-level update are not met; run a full update."""


async def _read_processed_text(raw_data_location: str) -> str:
    """Read the stored processed text file (pattern from TextDocument.read)."""
    async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
        return file.read()


async def _get_stored_chunks(document_id: UUID, old_text: str) -> List[dict]:
    """Return the document's stored chunk nodes (full props) in document order.

    Chunks are discovered via their ``is_part_of`` edges and ordered by where
    their text occurs in the stored processed text, so the ordering holds even
    after earlier incremental updates moved chunk positions.
    """
    graph_engine = await get_graph_engine()
    connections = await graph_engine.get_connections(str(document_id))
    # get_connections puts the queried node in the "source" slot regardless of
    # direction and omits large properties — take the true endpoints from the
    # edge and fetch full chunk nodes (including text) separately.
    chunk_ids = []
    for _source, edge, _target in connections:
        if "is_part_of" not in str(edge.get("relationship_name", "")):
            continue
        source_id = str(edge.get("source_node_id"))
        if str(edge.get("target_node_id")) == str(document_id) and source_id != str(document_id):
            chunk_ids.append(source_id)

    chunk_nodes = await graph_engine.get_nodes(chunk_ids) if chunk_ids else []
    chunks = [node for node in chunk_nodes if node.get("text") is not None]

    if not chunks:
        raise IncrementalUpdateNotPossible(
            f"document {document_id} has no stored chunks in the graph (not cognified yet?)"
        )

    def position(node):
        found = old_text.find(node["text"])
        if found < 0:
            raise IncrementalUpdateNotPossible(
                "stored chunk text not found in stored document text"
            )
        return found

    return sorted(chunks, key=position)


def _build_document(data: Data) -> TextDocument:
    """Mirror classify_documents' Document construction for this data row."""
    return TextDocument(
        id=data.id,
        title=f"{data.name}.{data.extension}",
        raw_data_location=data.raw_data_location,
        name=data.name,
        mime_type=data.mime_type,
        external_metadata=json.dumps(data.external_metadata, indent=4),
        importance_weight=data.importance_weight if data.importance_weight is not None else 0.5,
    )


def _build_new_chunks(
    document: TextDocument, plan: IncrementalPlan, stored_chunks: List[dict], word_size
) -> List[DocumentChunk]:
    """DocumentChunk objects for the replacement region.

    Ids follow the content-hash scheme used by TextChunker: occurrence counting
    runs over the final document order (kept prefix first), so two identical
    texts stay distinct. Surviving chunks created under an older id scheme are
    guarded against by bumping the occurrence on a direct id collision.
    """
    surviving_ids = {
        str(node["id"])
        for position, node in enumerate(stored_chunks)
        if position not in set(plan.affected_indices)
    }
    occurrences: dict = {}
    for position in range(plan.unchanged_prefix_count):
        text_hash = chunk_content_hash(stored_chunks[position]["text"])
        occurrences[text_hash] = occurrences.get(text_hash, 0) + 1

    chunks = []
    for offset, text in enumerate(plan.new_chunk_texts):
        content_hash = chunk_content_hash(text)
        occurrence = occurrences.get(content_hash, 0)
        chunk_id = content_chunk_id(document.id, content_hash, occurrence)
        while str(chunk_id) in surviving_ids:
            occurrence += 1
            chunk_id = content_chunk_id(document.id, content_hash, occurrence)
        occurrences[content_hash] = occurrence + 1
        chunks.append(
            DocumentChunk(
                id=chunk_id,
                text=text,
                chunk_size=sum(word_size(w) for w in text.split()),
                content_hash=content_hash,
                chunk_index=plan.unchanged_prefix_count + offset,
                cut_type="incremental_update",
                is_part_of=document,
                contains=[],
                document_id=str(document.id),
                document_name=document.name,
            )
        )
    return chunks


def _build_shifted_chunks(
    document: TextDocument,
    stored_chunks: List[dict],
    plan: IncrementalPlan,
    new_chunk_count: int,
) -> List[DocumentChunk]:
    """Surviving chunks whose chunk_index no longer matches their final position.

    Rebuilt with their EXISTING id and corrected index; re-storing them through
    add_data_points upserts the graph node and refreshes the vector payload, so
    citations and layout stay consistent after the region length changed.
    """
    total = len(stored_chunks)
    shifted = []
    for position, node in enumerate(stored_chunks):
        if position in set(plan.affected_indices):
            continue
        if position < plan.unchanged_prefix_count:
            expected = position
        else:
            offset_in_suffix = position - (total - plan.unchanged_suffix_count)
            expected = plan.unchanged_prefix_count + new_chunk_count + offset_in_suffix
        if int(node.get("chunk_index", -1)) == expected:
            continue
        text = node["text"]
        shifted.append(
            DocumentChunk(
                id=UUID(str(node["id"])),
                text=text,
                chunk_size=int(node.get("chunk_size", 0)),
                content_hash=node.get("content_hash") or chunk_content_hash(text),
                chunk_index=expected,
                cut_type=str(node.get("cut_type", "incremental_update")),
                is_part_of=document,
                contains=[],
                document_id=str(document.id),
                document_name=document.name,
            )
        )
    return shifted


async def _mark_document_processed(data_id: UUID, dataset_id: UUID) -> None:
    """Stamp cognify completion so a later cognify() doesn't redo the document."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        status_for_pipeline = data_point.pipeline_status.setdefault(PIPELINE_NAME, {})
        status_for_pipeline[str(dataset_id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
        await session.merge(data_point)
        await session.commit()


async def incremental_update(
    data_id: UUID,
    data,
    dataset_id: UUID,
    user: User,
    node_set: Optional[List[str]] = None,
    preferred_loaders=None,
) -> dict:
    """Perform a chunk-level incremental update of one document."""
    # -- Permissions: dataset write + membership + ownership ---------------- #
    dataset = await get_authorized_dataset(user, dataset_id, "write")
    dataset_data = await get_dataset_data(dataset.id)
    if not any(item.id == data_id for item in dataset_data):
        raise IncrementalUpdateNotPossible(f"data {data_id} is not part of dataset {dataset_id}")
    old_data = await get_data(user.id, data_id)  # raises on foreign data
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible("no stored processed text for this data item")

    # Same per-dataset lock as pipeline runs and delete_data: serialize against
    # concurrent cognify/delete/update on this dataset (re-entrant, so the
    # inner add() pipeline can take it again). Inside, establish the dataset's
    # database context — with backend access control on, graph/vector engines
    # resolve per user+dataset, and a fresh API request arrives without it.
    async with dataset_lock(dataset.id):
        async with set_database_global_context_variables(dataset.id, user.id):
            return await _run_incremental_update(
                data_id, data, dataset, user, old_data, node_set, preferred_loaders
            )


async def _run_incremental_update(
    data_id: UUID,
    data,
    dataset,
    user: User,
    old_data: Data,
    node_set: Optional[List[str]],
    preferred_loaders,
) -> dict:
    """The locked, dataset-context-scoped body of the incremental update."""
    dataset_id = dataset.id

    # -- Old state (must be captured BEFORE add() replaces the stored file) - #
    old_text = await _read_processed_text(old_data.raw_data_location)
    stored_chunks = await _get_stored_chunks(data_id, old_text)

    # -- New state through the standard add pipeline ------------------------ #
    # Wrapping in DataItem(data_id=...) routes ingest_data into its UPDATE
    # branch for this exact row (text input would otherwise mint a new
    # content-addressed id and leave the old row untouched).
    from cognee.tasks.ingestion.data_item import DataItem

    # Both caching layers must be off: the add-pipeline's incremental skip and
    # the data cache would otherwise drop the item (its id already reads as
    # processed) before ingest_data can rewrite the stored text.
    await add(
        data=DataItem(data=data, data_id=data_id),
        dataset_id=dataset_id,
        user=user,
        node_set=node_set,
        preferred_loaders=preferred_loaders,
        incremental_loading=False,
        data_cache=False,
    )
    new_data = await get_data(user.id, data_id)
    new_text = await _read_processed_text(new_data.raw_data_location)

    # -- Plan ---------------------------------------------------------------- #
    max_chunk_size = await get_max_chunk_tokens()
    from cognee.tasks.chunks.chunk_by_sentence import get_word_size

    try:
        plan = compute_incremental_plan(
            old_text,
            [node["text"] for node in stored_chunks],
            new_text,
            max_chunk_size,
            word_size=get_word_size,
        )
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error)) from error

    if not plan.affected_indices and not plan.new_chunk_texts:
        logger.info("incremental update: content unchanged, nothing to do")
        return {"status": "unchanged", "deleted_chunks": 0, "added_chunks": 0}

    # -- Delete affected chunks + summaries + chunk-orphaned entities -------- #
    affected_chunk_ids = [str(stored_chunks[i]["id"]) for i in plan.affected_indices]
    await delete_chunks_incremental(affected_chunk_ids)

    # -- Re-ingest only the replacement chunks ------------------------------- #
    document = _build_document(new_data)
    new_chunks = _build_new_chunks(document, plan, stored_chunks, get_word_size)

    context = PipelineContext(
        user=user,
        data_item=new_data,
        dataset=dataset,
        pipeline_run_id=uuid4(),
        pipeline_name=PIPELINE_NAME,
    )
    await extract_graph_from_data(new_chunks, KnowledgeGraph, ctx=context)
    summaries = await summarize_text(new_chunks)
    await add_data_points(summaries, ctx=context)

    # -- Renumber surviving chunks whose position shifted --------------------- #
    shifted_chunks = _build_shifted_chunks(document, stored_chunks, plan, len(new_chunks))
    if shifted_chunks:
        await add_data_points(shifted_chunks, ctx=context)

    await _mark_document_processed(data_id, dataset_id)

    logger.info(
        "incremental update: kept %d+%d chunks, deleted %d, added %d, reindexed %d",
        plan.unchanged_prefix_count,
        plan.unchanged_suffix_count,
        len(affected_chunk_ids),
        len(new_chunks),
        len(shifted_chunks),
    )
    return {
        "status": "incremental",
        "deleted_chunks": len(affected_chunk_ids),
        "added_chunks": len(new_chunks),
        "kept_chunks": plan.unchanged_prefix_count + plan.unchanged_suffix_count,
        "reindexed_chunks": len(shifted_chunks),
    }
