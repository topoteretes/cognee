"""Chunk-level incremental update: diff, delete affected chunks, re-ingest.

Flow (only runs from the update endpoint):

1. Enforce dataset-level "write" permission and data membership/ownership
   (same checks as the raw-document read endpoint).
2. Read the OLD processed text from ``Data.raw_data_location`` and the stored
   chunk nodes from the graph, ordered by their position in that text.
3. Run ``add()`` so the new file goes through the normal add pipeline (loaders
   store the new processed text and update the ``Data`` row).
4. Diff old vs new text; expand the edit to affected chunk boundaries; re-chunk
   the replacement region with the standard TextChunker (same sentence/paragraph
   boundary semantics and token budget as pipeline chunks).
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

from pydantic import BaseModel
from sqlalchemy import and_, delete, select

from cognee.api.v1.add import add
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.graph.models import Node
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.ontology.ontology_config import Config
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.modules.pipelines.operations.log_pipeline_run_complete import (
    log_pipeline_run_complete,
)
from cognee.modules.pipelines.operations.log_pipeline_run_error import log_pipeline_run_error
from cognee.modules.pipelines.operations.log_pipeline_run_start import log_pipeline_run_start
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.shared.utils import send_telemetry
from cognee.tasks.documents.classify_documents import update_node_set
from cognee.tasks.documents.extract_chunks_from_documents import update_document_token_count
from cognee.tasks.graph.detect_contradictions import detect_contradictions
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.llm.utils import get_max_chunk_tokens
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.incremental_chunking import (
    IncrementalPlan,
    IncrementalPlanError,
    compute_incremental_plan,
    validate_no_loss,
)
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_dataset, get_data
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.modules.graph.methods.delete_chunks_incremental import (
    delete_chunks_incremental,
    edge_endpoints,
)
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_data_item import DataItemStatus
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points

logger = get_logger("incremental_update")

PIPELINE_NAME = "cognify_pipeline"  # attribute to the cognify pipeline's status key
# Distinct run-record name: dashboards see incremental runs without touching
# the skip logic that keys on cognify_pipeline's per-item status.
RUN_PIPELINE_NAME = "incremental_update_pipeline"

# Graph adapters whose get_connections/get_nodes shapes the incremental path
# has been verified against (see edge_endpoints for the shape differences).
# Anything else falls back to the full update.
SUPPORTED_GRAPH_PROVIDERS = {"kuzu", "ladybug", "neo4j", "postgres"}


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
    # Adapters disagree on connection shapes (see edge_endpoints); resolve the
    # true endpoints, then fetch full chunk nodes (including text) separately —
    # connection triples may omit large properties.
    chunk_ids = []
    for source, edge, target in connections:
        if "is_part_of" not in str(edge.get("relationship_name", "")):
            continue
        source_id, target_id = edge_endpoints(source, edge, target)
        if target_id == str(document_id) and source_id != str(document_id):
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
    document = TextDocument(
        id=data.id,
        title=f"{data.name}.{data.extension}",
        raw_data_location=data.raw_data_location,
        name=data.name,
        mime_type=data.mime_type,
        external_metadata=json.dumps(data.external_metadata, indent=4),
        importance_weight=data.importance_weight if data.importance_weight is not None else 0.5,
    )
    update_node_set(document)  # NodeSet tagging parity with classify_documents
    return document


def _resolve_extraction_config() -> Config:
    """Ontology configuration exactly as cognify's get_default_tasks resolves it."""
    ontology_config = get_ontology_env_config()
    if (
        ontology_config.ontology_file_path
        and ontology_config.ontology_resolver
        and ontology_config.matching_strategy
    ):
        return {
            "ontology_config": {
                "ontology_resolver": get_ontology_resolver_from_env(**ontology_config.to_dict())
            }
        }
    return {"ontology_config": {"ontology_resolver": get_default_ontology_resolver()}}


async def _prune_ledger_rows(data_id: UUID, dataset_id: UUID, doomed_ids: List[str]) -> None:
    """Drop rollback-ledger rows for nodes the incremental delete removed."""
    if not doomed_ids:
        return
    slugs = [UUID(doomed) for doomed in doomed_ids]
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        await session.execute(
            delete(Node).where(
                and_(Node.data_id == data_id, Node.dataset_id == dataset_id, Node.slug.in_(slugs))
            )
        )
        await session.commit()


async def _chunk_region(
    document: TextDocument, region_text: str, max_chunk_size: int
) -> List[DocumentChunk]:
    """Run the standard TextChunker over the replacement region.

    Replacement chunks get the same sentence/paragraph boundary semantics as
    pipeline chunks; only the region's last chunk may come out under-filled,
    like the tail of any normally-cognified document. The chunker's ids and
    indexes are region-local and discarded — _build_new_chunks reassigns both.
    """
    if not region_text:
        return []

    async def get_text():
        yield region_text

    chunker = TextChunker(document, get_text, max_chunk_size)
    return [chunk async for chunk in chunker.read()]


def _build_new_chunks(
    document: TextDocument,
    plan: IncrementalPlan,
    stored_chunks: List[dict],
    region_chunks: List[DocumentChunk],
) -> List[DocumentChunk]:
    """Finalize the region's chunks with document-scoped identity and position.

    Ids follow the content-hash scheme: occurrence counting runs over the final
    document order (kept prefix first), so two identical texts stay distinct —
    the chunker's own region-local ids would collide with an identical prefix
    chunk. Surviving chunks created under an older id scheme are guarded
    against by bumping the occurrence on a direct id collision.
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
    for offset, region_chunk in enumerate(region_chunks):
        text = region_chunk.text
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
                chunk_size=region_chunk.chunk_size,
                content_hash=content_hash,
                chunk_index=plan.unchanged_prefix_count + offset,
                cut_type=region_chunk.cut_type,
                is_part_of=document,
                contains=[],
                document_id=str(document.id),
                document_name=document.name,
            )
        )
    return chunks


def _rehydrate_chunk(document: TextDocument, node: dict, chunk_index: int) -> DocumentChunk:
    """Rebuild a stored chunk at a new position, preserving every model field.

    Adapters replace the node's whole property set on MERGE (ladybug rewrites
    the properties blob wholesale), so any field missing here is not merely
    reset — it is erased. Everything DocumentChunk models must be carried over
    from the stored node; only the position (chunk_index) is meant to change.
    """
    text = node["text"]
    truth_alignment = node.get("truth_alignment")
    return DocumentChunk(
        id=UUID(str(node["id"])),
        text=text,
        chunk_size=int(node.get("chunk_size", 0)),
        content_hash=node.get("content_hash") or chunk_content_hash(text),
        chunk_index=chunk_index,
        cut_type=str(node.get("cut_type", "paragraph_end")),
        is_part_of=document,
        contains=[],
        importance_weight=node.get("importance_weight", document.importance_weight),
        document_id=str(document.id),
        document_name=document.name,
        truth_alignment=truth_alignment if isinstance(truth_alignment, list) else None,
        truth_epoch=node.get("truth_epoch"),
        ontology_valid=bool(node.get("ontology_valid", False)),
        ontology_uri=node.get("ontology_uri"),
        version=int(node.get("version", 1)),
        topological_rank=node.get("topological_rank", 0),
    )


def _build_shifted_chunks(
    document: TextDocument,
    stored_chunks: List[dict],
    plan: IncrementalPlan,
    new_chunk_count: int,
) -> List[DocumentChunk]:
    """Surviving chunks whose chunk_index no longer matches their final position.

    Rehydrated from their stored node with their EXISTING id and corrected
    index; re-storing them through add_data_points upserts the graph node and
    refreshes the vector payload, so citations and layout stay consistent
    after the region length changed.
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
        shifted.append(_rehydrate_chunk(document, node, expected))
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
    graph_model: type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
) -> dict:
    """Perform a chunk-level incremental update of one document."""
    graph_provider = str(get_graph_config().graph_database_provider).lower()
    if graph_provider not in SUPPORTED_GRAPH_PROVIDERS:
        raise IncrementalUpdateNotPossible(
            f"graph provider '{graph_provider}' is not verified for chunk-level updates"
        )

    # The HTTP router (and permissive SDK callers) send a list of files; a
    # chunk-level update targets exactly one document by definition.
    if isinstance(data, (list, tuple)):
        if len(data) != 1:
            raise IncrementalUpdateNotPossible(
                f"chunk-level update targets exactly one document, got {len(data)} items"
            )
        data = data[0]

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
                data_id,
                data,
                dataset,
                user,
                old_data,
                node_set,
                preferred_loaders,
                graph_model,
                custom_prompt,
            )


async def _run_incremental_update(
    data_id: UUID,
    data,
    dataset,
    user: User,
    old_data: Data,
    node_set: Optional[List[str]],
    preferred_loaders,
    graph_model: type[BaseModel],
    custom_prompt: Optional[str],
) -> dict:
    """Run-record-logged wrapper around the incremental update body."""
    pipeline_id = generate_pipeline_id(user.id, dataset.id, RUN_PIPELINE_NAME)
    pipeline_run = await log_pipeline_run_start(
        pipeline_id, RUN_PIPELINE_NAME, dataset.id, [data_id]
    )
    send_telemetry(
        "Incremental Update Run Started",
        user.id,
        additional_properties={"dataset_id": str(dataset.id), "data_id": str(data_id)},
    )
    try:
        result = await _apply_incremental_update(
            data_id,
            data,
            dataset,
            user,
            old_data,
            node_set,
            preferred_loaders,
            graph_model,
            custom_prompt,
        )
    except Exception as error:
        # Includes IncrementalUpdateNotPossible: the record shows why this run
        # ended and the full update that follows logs its own runs.
        await log_pipeline_run_error(
            pipeline_run.pipeline_run_id,
            pipeline_id,
            RUN_PIPELINE_NAME,
            dataset.id,
            [data_id],
            error,
        )
        raise
    await log_pipeline_run_complete(
        pipeline_run.pipeline_run_id, pipeline_id, RUN_PIPELINE_NAME, dataset.id, result
    )
    send_telemetry(
        "Incremental Update Run Completed",
        user.id,
        additional_properties={"dataset_id": str(dataset.id), "data_id": str(data_id), **result},
    )
    return result


async def _apply_incremental_update(
    data_id: UUID,
    data,
    dataset,
    user: User,
    old_data: Data,
    node_set: Optional[List[str]],
    preferred_loaders,
    graph_model: type[BaseModel],
    custom_prompt: Optional[str],
) -> dict:
    """The locked, dataset-context-scoped body of the incremental update."""
    dataset_id = dataset.id

    # Re-fetch the row INSIDE the lock: a concurrent update that just finished
    # has moved raw_data_location to a new processed file, and diffing against
    # the pre-lock snapshot would use a stale baseline.
    old_data = await get_data(user.id, data_id)
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible("data row disappeared before the update ran")

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
    stored_texts = [node["text"] for node in stored_chunks]
    try:
        plan = compute_incremental_plan(old_text, stored_texts, new_text)
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error)) from error

    document = _build_document(new_data)
    context = PipelineContext(
        user=user,
        data_item=new_data,
        dataset=dataset,
        pipeline_run_id=uuid4(),
        pipeline_name=PIPELINE_NAME,
    )

    if not plan.affected_indices:
        # Self-heal: a crash between an earlier delete and its renumbering can
        # leave stale indexes behind an unchanged text — repair them here.
        repaired = _build_shifted_chunks(document, stored_chunks, plan, 0)
        if repaired:
            await add_data_points(repaired, ctx=context)
        await _mark_document_processed(data_id, dataset_id)
        logger.info("incremental update: content unchanged, repaired %d indexes", len(repaired))
        return {
            "status": "unchanged",
            "deleted_chunks": 0,
            "added_chunks": 0,
            "reindexed_chunks": len(repaired),
        }

    # -- Chunk the region with the standard TextChunker, then verify no-loss -- #
    max_chunk_size = await get_max_chunk_tokens()
    region_chunks = await _chunk_region(document, plan.replacement_region, max_chunk_size)
    new_chunks = _build_new_chunks(document, plan, stored_chunks, region_chunks)
    try:
        validate_no_loss(stored_texts, plan, [chunk.text for chunk in new_chunks], new_text)
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error)) from error

    # A replacement chunk that is byte-identical to one being replaced hashes
    # to the SAME node id. Keep its subgraph: rehydrate the stored node at its
    # new position (preserving all properties), skip re-extraction, and
    # exclude it from deletion.
    affected_ids = {str(stored_chunks[i]["id"]) for i in plan.affected_indices}
    new_ids = {str(chunk.id) for chunk in new_chunks}
    stored_by_id = {str(node["id"]): node for node in stored_chunks}
    reused_chunks = [
        _rehydrate_chunk(document, stored_by_id[str(chunk.id)], chunk.chunk_index)
        for chunk in new_chunks
        if str(chunk.id) in affected_ids
    ]
    fresh_chunks = [chunk for chunk in new_chunks if str(chunk.id) not in affected_ids]

    # -- Ingest new content FIRST (crash safety: a failure between phases ----- #
    #    leaves recoverable duplicates, never holes; the retry falls back to a
    #    full update because the stored chunks no longer tile the stored text).
    cognify_config = get_cognify_config()
    if fresh_chunks:
        # Same extraction + summarization the cognify pipeline runs, with the
        # same ontology resolution, model, and prompt plumbing.
        summaries = await extract_graph_and_summarize(
            fresh_chunks,
            graph_model=graph_model,
            config=_resolve_extraction_config(),
            custom_prompt=custom_prompt,
            ctx=context,
        )
        await add_data_points(
            summaries, ctx=context, embed_triplets=cognify_config.triplet_embedding
        )
        if cognify_config.contradiction_detection:
            await detect_contradictions(summaries)
    if reused_chunks:
        await add_data_points(reused_chunks, ctx=context)

    # -- Delete replaced chunks + summaries + chunk-orphaned entities --------- #
    ids_to_delete = sorted(affected_ids - new_ids)
    if ids_to_delete:
        doomed = await delete_chunks_incremental(ids_to_delete)
        await _prune_ledger_rows(data_id, dataset_id, doomed)

    # -- Renumber surviving chunks whose position shifted --------------------- #
    shifted_chunks = _build_shifted_chunks(document, stored_chunks, plan, len(new_chunks))
    if shifted_chunks:
        await add_data_points(shifted_chunks, ctx=context)

    # -- Keep Data.token_count in sync with the final chunk set --------------- #
    surviving_tokens = sum(
        int(node.get("chunk_size", 0))
        for position, node in enumerate(stored_chunks)
        if position not in set(plan.affected_indices)
    )
    new_tokens = sum(chunk.chunk_size for chunk in new_chunks)
    await update_document_token_count(data_id, surviving_tokens + new_tokens)

    await _mark_document_processed(data_id, dataset_id)

    logger.info(
        "incremental update: kept %d+%d chunks, deleted %d, added %d (%d reused), reindexed %d",
        plan.unchanged_prefix_count,
        plan.unchanged_suffix_count,
        len(ids_to_delete),
        len(new_chunks),
        len(reused_chunks),
        len(shifted_chunks),
    )
    return {
        "status": "incremental",
        "deleted_chunks": len(ids_to_delete),
        "added_chunks": len(new_chunks),
        "reused_chunks": len(reused_chunks),
        "kept_chunks": plan.unchanged_prefix_count + plan.unchanged_suffix_count,
        "reindexed_chunks": len(shifted_chunks),
    }
