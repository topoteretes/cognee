"""Chunk-level incremental update: stage, plan, validate, write, publish.

Flow (only runs from the update endpoint):

1. Enforce dataset-level "write" permission and data membership/ownership
   (same checks as the raw-document read endpoint).
2. Read the OLD processed text from ``Data.raw_data_location`` and the stored
   chunk nodes from the graph, ordered by their position in that text.
3. STAGE the new content: the input runs through the same loader machinery as
   ingestion (content-addressed storage write) — but the ``Data`` row is NOT
   touched. Readers keep resolving the coherent old version.
4. Diff old vs new text into DISJOINT changed regions (line-anchored hunks,
   trimmed to char precision, expanded to old chunk boundaries); chunks between
   regions are kept untouched. Each region is re-chunked with the standard
   TextChunker (same boundary semantics and token budget as pipeline chunks),
   then ``validate_no_loss`` proves the final chunk set reassembles the new
   text byte-for-byte. Only now — staging and validation done — is a pipeline
   run record created; refused updates leave no run-record noise.
5. Write: extract ONLY the new chunks through the standard graph-extraction
   and storage tasks (attributed to the same ``data_id``), delete the replaced
   chunks (+ summaries + chunk-orphaned entities), renumber shifted survivors.
6. PUBLISH in one relational transaction: content location, hashes, size,
   token count, and the processed stamp flip together. A crash anywhere
   before the publish leaves the row on the old content; the stored chunks
   then no longer tile the stored text, so the next touch fails closed into a
   full rebuild (self-heal).

Raises IncrementalUpdateNotPossible when preconditions fail (first ingestion,
non-text data, stored chunks not tiling the stored text) — the caller decides
to run the full update instead.
"""

import json
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import and_, delete, select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.vector import get_vector_engine_async
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
from cognee.modules.ingestion import classify
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage
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

    Chunks are discovered via their ``is_part_of`` edges and ordered by their
    stored ``chunk_index`` — authoritative, because ingestion assigns it
    sequentially and every incremental update renumbers survivors (with a
    self-heal pass for interrupted runs). Ordering by text position instead
    breaks on documents with repeated content: a later chunk's text can be
    found at an earlier occurrence, scrambling the order and forcing a
    needless full-update fallback. The tiling check in the planner remains
    the correctness gate for any document whose indexes are stale.
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

    return sorted(chunks, key=lambda node: int(node.get("chunk_index", -1)))


def _build_document(data: Data, staged: Optional["StagedContent"] = None) -> TextDocument:
    """Mirror classify_documents' Document construction for this data row.

    With ``staged`` the document describes the NEW content (name, location,
    mime type from the staged files) under the row's stable id — the chunks
    written during the update must carry post-publish metadata even though
    the row itself flips only at the end.
    """
    name = staged.name if staged else data.name
    extension = staged.extension if staged else data.extension
    document = TextDocument(
        id=data.id,
        title=f"{name}.{extension}",
        raw_data_location=staged.raw_data_location if staged else data.raw_data_location,
        name=name,
        mime_type=staged.mime_type if staged else data.mime_type,
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


def _assemble_final_chunks(
    document: TextDocument,
    stored_chunks: List[dict],
    plan: IncrementalPlan,
    region_chunk_lists: List[List[DocumentChunk]],
) -> tuple:
    """Walk the final document order once: kept chunks and region chunks interleaved.

    Returns (new_chunks, kept_final_index) where new_chunks carry document-
    scoped content-hash ids (occurrence counting over the FINAL order, so two
    identical texts stay distinct; surviving legacy ids are dodged by bumping
    the occurrence) and kept_final_index maps each kept chunk's old position
    to its final chunk_index.
    """
    affected = set(plan.affected_indices)
    surviving_ids = {
        str(node["id"]) for position, node in enumerate(stored_chunks) if position not in affected
    }
    region_by_start = {
        region.affected_indices[0]: index for index, region in enumerate(plan.regions)
    }

    occurrences: dict = {}
    new_chunks: List[DocumentChunk] = []
    kept_final_index: dict = {}
    final_index = 0
    position = 0
    while position < len(stored_chunks):
        if position in region_by_start:
            region_index = region_by_start[position]
            for region_chunk in region_chunk_lists[region_index]:
                text = region_chunk.text
                content_hash = chunk_content_hash(text)
                occurrence = occurrences.get(content_hash, 0)
                chunk_id = content_chunk_id(document.id, content_hash, occurrence)
                while str(chunk_id) in surviving_ids:
                    occurrence += 1
                    chunk_id = content_chunk_id(document.id, content_hash, occurrence)
                occurrences[content_hash] = occurrence + 1
                new_chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=text,
                        chunk_size=region_chunk.chunk_size,
                        content_hash=content_hash,
                        max_chunk_tokens=region_chunk.max_chunk_tokens,
                        chunk_index=final_index,
                        cut_type=region_chunk.cut_type,
                        is_part_of=document,
                        contains=[],
                        document_id=str(document.id),
                        document_name=document.name,
                    )
                )
                final_index += 1
            position = plan.regions[region_index].affected_indices[-1] + 1
        else:
            text_hash = chunk_content_hash(stored_chunks[position]["text"])
            occurrences[text_hash] = occurrences.get(text_hash, 0) + 1
            kept_final_index[position] = final_index
            final_index += 1
            position += 1
    return new_chunks, kept_final_index


def _region_chunk_budget(stored_chunks: List[dict], region, fallback: int) -> int:
    """Token budget for re-chunking one region.

    The budget recorded on the chunks the region replaces wins, so an edit
    keeps the granularity of the text around it even when the global
    configuration changed after ingestion. Legacy chunks predate the recorded
    budget and fall back to the current configuration.
    """
    for position in region.affected_indices:
        recorded = stored_chunks[position].get("max_chunk_tokens")
        if recorded:
            return int(recorded)
    return fallback


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
        max_chunk_tokens=node.get("max_chunk_tokens"),
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
    affected: set,
    kept_final_index: dict,
) -> List[DocumentChunk]:
    """Kept chunks whose chunk_index no longer matches their final position.

    Rehydrated from their stored node with their EXISTING id and corrected
    index; re-storing them through add_data_points upserts the graph node and
    refreshes the vector payload, so citations and layout stay consistent
    after region lengths changed.
    """
    shifted = []
    for position, node in enumerate(stored_chunks):
        if position in affected:
            continue
        expected = kept_final_index[position]
        if int(node.get("chunk_index", -1)) == expected:
            continue
        shifted.append(_rehydrate_chunk(document, node, expected))
    return shifted


async def _restore_repositioned_chunks(chunks: List[DocumentChunk], context) -> None:
    """Write back chunks whose ONLY change is their position (kept or reused).

    Graph node properties refresh through the normal MERGE path; the vector
    row takes a payload-only update where the adapter supports it — the text
    is unchanged, so re-embedding it would be pure waste (an early edit in a
    large document repositions O(document) chunks). Adapters without
    update_payload take the full re-embedding write: same stored state,
    higher cost.
    """
    vector_engine = await get_vector_engine_async()
    if not getattr(vector_engine, "supports_payload_update", False):
        await add_data_points(chunks, ctx=context)
        return
    await add_data_points(chunks, ctx=context, graph_only=True)
    # Only chunk_index changed; content_hash is deliberately not written here —
    # legacy collections may predate the field in their payload schema.
    await vector_engine.update_payload(
        "DocumentChunk_text",
        {str(chunk.id): {"chunk_index": chunk.chunk_index} for chunk in chunks},
    )


class StagedContent(BaseModel):
    """The new content, processed and stored — with the Data row untouched.

    Content-addressed storage writes are safe to make before anything is
    decided: until ``_publish_staged`` flips the row, readers keep resolving
    the old file, and an abandoned staged file is inert garbage.
    """

    name: str
    raw_data_location: str
    original_data_location: str
    extension: str
    mime_type: str
    original_extension: str
    original_mime_type: str
    loader_engine: str
    content_hash: str
    raw_content_hash: str
    data_size: int


async def _stage_new_content(data, preferred_loaders) -> StagedContent:
    """Run the input through ingestion's loader machinery without row writes.

    Mirrors ingest_data's per-item processing exactly (same storage layout,
    same metadata derivation), so the published row is indistinguishable from
    one written by a full ingestion.
    """
    original_file_path = await save_data_item_to_storage(data)
    actual_file_path = get_data_file_path(original_file_path)

    storage_file_path, loader_engine = await data_item_to_text_file(
        actual_file_path, preferred_loaders
    )
    if loader_engine is None:
        raise IncrementalUpdateNotPossible("no loader accepted the new content")

    async with open_data_file(original_file_path) as file:
        original_metadata = classify(file).get_metadata()
    async with open_data_file(storage_file_path) as file:
        storage_metadata = classify(file).get_metadata()

    return StagedContent(
        name=original_metadata["name"],
        raw_data_location=storage_file_path,
        original_data_location=original_metadata["file_path"],
        extension=storage_metadata["extension"],
        mime_type=storage_metadata["mime_type"],
        original_extension=original_metadata["extension"],
        original_mime_type=original_metadata["mime_type"],
        loader_engine=loader_engine.loader_name,
        content_hash=original_metadata["content_hash"],
        raw_content_hash=storage_metadata["content_hash"],
        data_size=original_metadata["file_size"],
    )


async def _publish_staged(
    data_id: UUID,
    dataset_id: UUID,
    staged: StagedContent,
    token_count: int,
    node_set: Optional[List[str]],
) -> None:
    """The one-transaction publish: content, metadata, and status flip together.

    Everything readers can observe about the document — stored text location,
    hashes, size, token count, and the cognify-completed stamp — commits
    atomically. Any crash before this leaves the row entirely on the old
    version.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            raise IncrementalUpdateNotPossible("data row disappeared before publish")

        data_point.name = staged.name
        data_point.raw_data_location = staged.raw_data_location
        data_point.original_data_location = staged.original_data_location
        data_point.extension = staged.extension
        data_point.mime_type = staged.mime_type
        data_point.original_extension = staged.original_extension
        data_point.original_mime_type = staged.original_mime_type
        data_point.loader_engine = staged.loader_engine
        data_point.content_hash = staged.content_hash
        data_point.raw_content_hash = staged.raw_content_hash
        data_point.data_size = staged.data_size
        data_point.token_count = token_count
        if node_set:
            data_point.node_set = json.dumps(node_set)

        status_for_pipeline = data_point.pipeline_status.setdefault(PIPELINE_NAME, {})
        status_for_pipeline[str(dataset_id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED

        await session.merge(data_point)
        await session.commit()


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
    old_data = await get_data(user.id, data_id, dataset.id)  # raises on foreign data
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
    """Stage → validate → (record) → write → publish.

    Run-record discipline: no PipelineRun exists until staging and validation
    have succeeded — a refused or unchanged update leaves no run-record noise.
    Errors after the record is created are logged against it.
    """
    bundle = await _stage_and_plan(data_id, data, dataset, user, node_set, preferred_loaders)

    if bundle.get("status") == "unchanged":
        return bundle

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
        result = await _write_and_publish(
            bundle, data_id, dataset, user, node_set, graph_model, custom_prompt
        )
    except Exception as error:
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


async def _stage_and_plan(
    data_id: UUID,
    data,
    dataset,
    user: User,
    node_set: Optional[List[str]],
    preferred_loaders,
) -> dict:
    """Everything that can be decided WITHOUT touching live state.

    Stages the new content (content-addressed storage write, row untouched),
    reads the old state, plans the diff, chunks the regions at their recorded
    budgets, and proves no-loss. Raises IncrementalUpdateNotPossible freely —
    at this point nothing observable has changed and no run record exists.
    """
    dataset_id = dataset.id

    # Re-fetch the row INSIDE the lock: a concurrent update that just finished
    # has moved raw_data_location to a new processed file, and diffing against
    # the pre-lock snapshot would use a stale baseline.
    old_data = await get_data(user.id, data_id, dataset_id)
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible("data row disappeared before the update ran")

    old_text = await _read_processed_text(old_data.raw_data_location)
    stored_chunks = await _get_stored_chunks(data_id, old_text)

    # Stage the new content through ingestion's own loader machinery. The Data
    # row is NOT written: readers keep resolving the old version until publish.
    staged = await _stage_new_content(data, preferred_loaders)
    new_text = await _read_processed_text(staged.raw_data_location)

    document = _build_document(old_data, staged)

    if staged.content_hash == old_data.content_hash and new_text == old_text:
        # Same content re-submitted. Self-heal: a crash between an earlier
        # delete and its renumbering can leave stale indexes — repair them.
        context = PipelineContext(
            user=user,
            data_item=old_data,
            dataset=dataset,
            pipeline_run_id=uuid4(),
            pipeline_name=PIPELINE_NAME,
        )
        repaired = _build_shifted_chunks(
            document, stored_chunks, set(), {i: i for i in range(len(stored_chunks))}
        )
        if repaired:
            await _restore_repositioned_chunks(repaired, context)
        await _mark_document_processed(data_id, dataset_id)
        logger.info("incremental update: content unchanged, repaired %d indexes", len(repaired))
        return {
            "status": "unchanged",
            "deleted_chunks": 0,
            "added_chunks": 0,
            "reused_chunks": 0,
            "reindexed_chunks": len(repaired),
        }

    stored_texts = [node["text"] for node in stored_chunks]
    try:
        plan = compute_incremental_plan(old_text, stored_texts, new_text)
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error)) from error

    fallback_budget = await get_max_chunk_tokens()
    region_chunk_lists = [
        await _chunk_region(
            document,
            region.replacement_text,
            _region_chunk_budget(stored_chunks, region, fallback_budget),
        )
        for region in plan.regions
    ]
    new_chunks, kept_final_index = _assemble_final_chunks(
        document, stored_chunks, plan, region_chunk_lists
    )
    try:
        validate_no_loss(
            stored_texts,
            plan,
            [[chunk.text for chunk in chunks] for chunks in region_chunk_lists],
            new_text,
        )
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error)) from error

    return {
        "staged": staged,
        "document": document,
        "data_item": old_data,
        "stored_chunks": stored_chunks,
        "plan": plan,
        "new_chunks": new_chunks,
        "kept_final_index": kept_final_index,
    }


async def _write_and_publish(
    bundle: dict,
    data_id: UUID,
    dataset,
    user: User,
    node_set: Optional[List[str]],
    graph_model: type[BaseModel],
    custom_prompt: Optional[str],
) -> dict:
    """The write phase, ending in the one-transaction publish.

    Order is crash-shaped: fresh content lands first (a failure leaves
    recoverable duplicates, never holes), replaced chunks are deleted, shifted
    survivors renumber — and only then does the row flip to the new content,
    hashes, token count, and completed stamp in a single transaction. Any
    crash before the flip leaves readers on the coherent old version; the
    stored chunks then fail the tiling gate on the next touch and the
    document self-heals via full rebuild.
    """
    dataset_id = dataset.id
    staged: StagedContent = bundle["staged"]
    document = bundle["document"]
    stored_chunks = bundle["stored_chunks"]
    plan = bundle["plan"]
    new_chunks = bundle["new_chunks"]
    kept_final_index = bundle["kept_final_index"]

    context = PipelineContext(
        user=user,
        data_item=bundle.get("data_item"),
        dataset=dataset,
        pipeline_run_id=uuid4(),
        pipeline_name=PIPELINE_NAME,
    )

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
        await _restore_repositioned_chunks(reused_chunks, context)

    # -- Delete replaced chunks + summaries + chunk-orphaned entities --------- #
    ids_to_delete = sorted(affected_ids - new_ids)
    if ids_to_delete:
        doomed = await delete_chunks_incremental(ids_to_delete)
        await _prune_ledger_rows(data_id, dataset_id, doomed)

    # -- Renumber kept chunks whose position shifted --------------------------- #
    shifted_chunks = _build_shifted_chunks(
        document, stored_chunks, set(plan.affected_indices), kept_final_index
    )
    if shifted_chunks:
        await _restore_repositioned_chunks(shifted_chunks, context)

    # -- PUBLISH: content + metadata + token count + stamp, atomically --------- #
    surviving_tokens = sum(
        int(node.get("chunk_size", 0))
        for position, node in enumerate(stored_chunks)
        if position not in set(plan.affected_indices)
    )
    new_tokens = sum(chunk.chunk_size for chunk in new_chunks)
    await _publish_staged(data_id, dataset_id, staged, surviving_tokens + new_tokens, node_set)

    kept_count = len(stored_chunks) - len(plan.affected_indices)
    logger.info(
        "incremental update: %d regions, kept %d chunks, deleted %d, added %d "
        "(%d reused), reindexed %d",
        len(plan.regions),
        kept_count,
        len(ids_to_delete),
        len(new_chunks),
        len(reused_chunks),
        len(shifted_chunks),
    )
    return {
        "status": "incremental",
        "regions": len(plan.regions),
        "deleted_chunks": len(ids_to_delete),
        "added_chunks": len(new_chunks),
        "reused_chunks": len(reused_chunks),
        "kept_chunks": kept_count,
        "reindexed_chunks": len(shifted_chunks),
    }
