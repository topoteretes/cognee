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
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import and_, delete, select

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
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
from cognee.modules.chunking.chunk_id import chunk_content_hash
from cognee.modules.chunking.chunk_policy import (
    DEFAULT_CHUNK_POLICY,
    ChunkPlan,
    ChunkPlanRequest,
    ChunkPolicy,
    IncrementalPlanError,
    stored_chunker_id,
)
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_dataset, get_data
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.tasks.documents.classify_documents import document_class_for
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


class RefusalReason(str, Enum):
    """Why a chunk-level update fell back to the full flow.

    Every refusal used to surface as one free-text message and one log line, so
    a permanent misconfiguration (an incompatible chunker, an unsupported
    backend) looked exactly like a first ingestion. The reason is logged as a
    structured field so they are separable.
    """

    UNSUPPORTED_BACKEND = "unsupported_backend"
    UNSUPPORTED_CHUNKER = "unsupported_chunker"
    NO_BASELINE = "no_baseline"
    CHUNKS_NOT_TILING = "chunks_not_tiling"
    UNREADABLE_TEXT = "unreadable_text"


class IncrementalUpdateNotPossible(Exception):
    """Preconditions for a chunk-level update are not met; run a full update."""

    def __init__(self, message: str, reason: RefusalReason = RefusalReason.NO_BASELINE):
        super().__init__(message)
        self.reason = reason


async def _read_processed_text(raw_data_location: str) -> str:
    """Read the stored processed text file (pattern from TextDocument.read).

    A row still pointing at binary content (pre-0.3.7 ingestion) is a refusal,
    not a 500: the full rebuild handles it. Supporting those rows on this path
    stays out of scope; falling back instead of failing does not.
    """
    try:
        async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError as error:
        raise IncrementalUpdateNotPossible(
            f"stored text at {raw_data_location} is not valid UTF-8",
            RefusalReason.UNREADABLE_TEXT,
        ) from error


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
            f"document {document_id} has no stored chunks in the graph (not cognified yet?)",
            RefusalReason.NO_BASELINE,
        )

    return sorted(chunks, key=lambda node: int(node.get("chunk_index", -1)))


def _merged_external_metadata(data: Data, node_set: Optional[List[str]]) -> dict:
    """The row's external metadata with an explicitly supplied node_set applied.

    Mirrors ``ingest_data``'s ``ext_metadata["node_set"] = node_set`` so the
    incremental path tags new chunks with the node_set the CALLER passed rather
    than the one stored before the update. ``_publish_staged`` writes the same
    merged dict back, keeping the chunks, ``Data.node_set`` and
    ``Data.external_metadata`` on one value instead of three.
    """
    metadata = dict(data.external_metadata or {})
    if node_set:
        metadata["node_set"] = node_set
    return metadata


def _build_document(
    data: Data,
    staged: Optional["StagedContent"] = None,
    node_set: Optional[List[str]] = None,
) -> Document:
    """Mirror classify_documents' Document construction for this data row.

    The class comes from ``document_class_for`` — the same dispatch
    classify_documents and cognify routing use — so an update never rewrites a
    row's document type to ``text``.

    With ``staged`` the document describes the NEW content (name, location,
    mime type from the staged files) under the row's stable id — the chunks
    written during the update must carry post-publish metadata even though
    the row itself flips only at the end.
    """
    name = staged.name if staged else data.name
    extension = staged.extension if staged else data.extension
    document_class = document_class_for(data)
    document = document_class(
        id=data.id,
        title=f"{name}.{extension}",
        raw_data_location=staged.raw_data_location if staged else data.raw_data_location,
        name=name,
        mime_type=staged.mime_type if staged else data.mime_type,
        external_metadata=json.dumps(_merged_external_metadata(data, node_set), indent=4),
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


def _rehydrate_chunk(document: Document, node: dict, chunk_index: int) -> DocumentChunk:
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
    document: Document,
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

    Fast path (both stores support narrow moves): the graph adapter's
    ``update_chunk_index`` patches ONLY the index — the stored node is the
    source of truth and every other property is carried verbatim, so nothing
    a rehydrated model forgets to declare can be erased — and the vector row
    takes a payload-only update (text unchanged, so re-embedding would be
    pure waste; an early edit in a large document repositions O(document)
    chunks). Where either store lacks its narrow operation, the full MERGE
    rewrite of rehydrated models remains: same stored state, higher cost and
    the carry-list burden.
    """
    from cognee.infrastructure.databases.exceptions import UnsupportedGraphOperation

    vector_engine = await get_vector_engine_async()
    supports_payload = getattr(vector_engine, "supports_payload_update", False)

    if supports_payload:
        graph_engine = await get_graph_engine()
        try:
            await graph_engine.update_chunk_index(
                {str(chunk.id): chunk.chunk_index for chunk in chunks}
            )
        except UnsupportedGraphOperation:
            await add_data_points(chunks, ctx=context, graph_only=True)
        # Only chunk_index changed; content_hash is deliberately not written
        # here — legacy collections may predate the field in their payload
        # schema.
        await vector_engine.update_payload(
            "DocumentChunk_text",
            {str(chunk.id): {"chunk_index": chunk.chunk_index} for chunk in chunks},
        )
        return

    await add_data_points(chunks, ctx=context)


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
        raise IncrementalUpdateNotPossible(
            "no loader accepted the new content", RefusalReason.UNREADABLE_TEXT
        )

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
            # Both copies move together: the column readers filter on, and the
            # metadata a later full cognify rebuilds its NodeSet tags from.
            # Writing only the column would leave a document whose chunks and
            # whose row disagree about their grouping.
            data_point.node_set = json.dumps(node_set)
            data_point.external_metadata = _merged_external_metadata(data_point, node_set)

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


async def _is_document_processed(data_id: UUID, dataset_id: UUID) -> bool:
    """Whether the cognify-completion stamp is already on the row."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()
        if data_point is None:
            return False
        return (data_point.pipeline_status or {}).get(PIPELINE_NAME, {}).get(
            str(dataset_id)
        ) == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED


def _unchanged_result(reindexed: int) -> dict:
    """The no-op result, shaped like the incremental one.

    The router returns this dict verbatim as the HTTP body, and both the SDK
    docstring and the route documentation advertise the same keys for either
    status — so a client reading kept_chunks must not get a KeyError on a no-op.
    """
    return {
        "status": "unchanged",
        "regions": 0,
        "deleted_chunks": 0,
        "added_chunks": 0,
        "reused_chunks": 0,
        "kept_chunks": 0,
        "reindexed_chunks": reindexed,
    }


async def _repair_unchanged(
    bundle: dict,
    data_id: UUID,
    dataset,
    user: User,
    pipeline_run_id: UUID,
) -> dict:
    """Execute the self-heal the planning phase found, under the caller's run."""
    shifted = bundle.get("shifted_chunks") or []
    if shifted:
        context = PipelineContext(
            user=user,
            data_item=bundle.get("data_item"),
            dataset=dataset,
            pipeline_run_id=pipeline_run_id,
            pipeline_name=PIPELINE_NAME,
        )
        await _restore_repositioned_chunks(shifted, context)
    await _mark_document_processed(data_id, dataset.id)
    logger.info(
        "incremental update: content unchanged, repaired %s",
        ", ".join(bundle.get("repairs") or ["nothing"]),
    )
    return _unchanged_result(len(shifted))


async def incremental_update(
    data_id: UUID,
    data,
    dataset_id: UUID,
    user: User,
    node_set: Optional[List[str]] = None,
    preferred_loaders=None,
    graph_model: type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
    chunker: type = TextChunker,
    policy: ChunkPolicy = DEFAULT_CHUNK_POLICY,
) -> dict:
    """Perform a chunk-level incremental update of one document.

    ``policy`` decides which chunks exist afterwards and what happens to the
    old ones; it is replaceable without touching storage or this orchestration.
    ``chunker`` must match the one that built the document's stored chunks —
    a mismatch is refused rather than discovered as a tiling failure.
    """
    graph_engine = await get_graph_engine()
    if not getattr(graph_engine, "supports_incremental_chunk_updates", False):
        raise IncrementalUpdateNotPossible(
            f"graph backend {type(graph_engine).__name__} does not support chunk-level updates",
            RefusalReason.UNSUPPORTED_BACKEND,
        )

    # Single-item input is update()'s contract, enforced there with
    # IngestionError. Re-checking it here would be unreachable from update()
    # and, worse, would raise IncrementalUpdateNotPossible — which means "fall
    # back", routing a multi-item update into a full flow that has already
    # unwrapped it.

    # -- Permissions: dataset write + delete + membership + ownership -------- #
    # Delete as well as write: this path removes chunks, summaries,
    # chunk-orphaned entities, orphaned entity types and triplet embeddings.
    # The full fallback path reaches the same destruction through
    # datasets.delete_data, which requires "delete" — without this check the
    # permission demanded by update() would depend on which branch it happened
    # to take, and the faster branch would be the weaker one. Raise the same
    # exception type delete_data raises for the same denial.
    try:
        dataset = await get_authorized_dataset(user, dataset_id, "write")
        await get_authorized_dataset(user, dataset_id, "delete")
    except PermissionDeniedError:
        raise UnauthorizedDataAccessError(f"Dataset {dataset_id} not accessible.")
    dataset_data = await get_dataset_data(dataset.id)
    if not any(item.id == data_id for item in dataset_data):
        raise IncrementalUpdateNotPossible(
            f"data {data_id} is not part of dataset {dataset_id}", RefusalReason.NO_BASELINE
        )
    old_data = await get_data(user.id, data_id, dataset.id)  # raises on foreign data
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible(
            "no stored processed text for this data item", RefusalReason.NO_BASELINE
        )

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
                chunker,
                policy,
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
    chunker: type,
    policy: ChunkPolicy,
) -> dict:
    """Stage → validate → (record) → write → publish.

    Run-record discipline: no PipelineRun exists until there is something to
    WRITE — a refused or genuinely no-op update leaves no run-record noise.
    Errors after the record is created are logged against it. Every write,
    including the unchanged branch's self-heal, happens under a run.
    """
    bundle = await _stage_and_plan(
        data_id, data, dataset, user, node_set, preferred_loaders, chunker, policy
    )

    # A no-op with nothing to repair is the only path that writes nothing, and
    # so the only one that records no run.
    if bundle.get("status") == "unchanged" and not bundle.get("repairs"):
        return _unchanged_result(0)

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
        if bundle.get("status") == "unchanged":
            result = await _repair_unchanged(
                bundle, data_id, dataset, user, pipeline_run.pipeline_run_id
            )
        else:
            result = await _write_and_publish(
                bundle,
                data_id,
                dataset,
                user,
                node_set,
                graph_model,
                custom_prompt,
                pipeline_run.pipeline_run_id,
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
    chunker: type,
    policy: ChunkPolicy,
) -> dict:
    """Everything that can be decided WITHOUT touching live state.

    Stages the new content (content-addressed storage write, row untouched),
    reads the old state, plans the diff, chunks the regions at their recorded
    budgets, and proves no-loss. Raises IncrementalUpdateNotPossible freely —
    at this point nothing observable has changed and no run record exists.

    This holds for the unchanged branch too: it reports the repairs it found
    and performs none of them. The caller opens a run and executes them, so a
    self-heal that fails is recorded rather than invisible.
    """
    dataset_id = dataset.id

    # Re-fetch the row INSIDE the lock: a concurrent update that just finished
    # has moved raw_data_location to a new processed file, and diffing against
    # the pre-lock snapshot would use a stale baseline.
    old_data = await get_data(user.id, data_id, dataset_id)
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible(
            "data row disappeared before the update ran", RefusalReason.NO_BASELINE
        )

    old_text = await _read_processed_text(old_data.raw_data_location)
    stored_chunks = await _get_stored_chunks(data_id, old_text)

    # Stage the new content through ingestion's own loader machinery. The Data
    # row is NOT written: readers keep resolving the old version until publish.
    staged = await _stage_new_content(data, preferred_loaders)
    new_text = await _read_processed_text(staged.raw_data_location)

    document = _build_document(old_data, staged, node_set)

    if staged.content_hash == old_data.content_hash and new_text == old_text:
        # Same content re-submitted. Two things can still be stale, and both
        # are reported rather than repaired here: chunk indexes left behind by
        # a crash between an earlier delete and its renumbering, and a missing
        # cognify-completion stamp (without which a later cognify would redo
        # the whole document). A missing stamp counts as a repair so the
        # self-heal keeps its reach — it is a write either way, and every write
        # belongs under a run.
        shifted = _build_shifted_chunks(
            document, stored_chunks, set(), {i: i for i in range(len(stored_chunks))}
        )
        needs_stamp = not await _is_document_processed(data_id, dataset_id)
        repairs = ["indexes"] if shifted else []
        if needs_stamp:
            repairs.append("stamp")
        return {
            "status": "unchanged",
            "repairs": repairs,
            "data_item": old_data,
            "shifted_chunks": shifted,
        }

    # Compatibility is a planning question, so answer it before planning. Every
    # chunker cuts differently — an overlapping one's output cannot tile its
    # input — and without this the mismatch would surface as a tiling failure,
    # indistinguishable from a document that was never cognified.
    stored_chunker = stored_chunker_id(stored_chunks)
    if stored_chunker and stored_chunker != getattr(chunker, "chunker_id", ""):
        raise IncrementalUpdateNotPossible(
            f"document was chunked by {stored_chunker}, not {getattr(chunker, 'chunker_id', '')}",
            RefusalReason.UNSUPPORTED_CHUNKER,
        )

    try:
        plan = await policy(
            ChunkPlanRequest(
                old_text=old_text,
                new_text=new_text,
                stored_chunks=stored_chunks,
                document=document,
                chunker_cls=chunker,
                fallback_budget=await get_max_chunk_tokens(),
            )
        )
    except IncrementalPlanError as error:
        raise IncrementalUpdateNotPossible(str(error), RefusalReason.CHUNKS_NOT_TILING) from error

    # Re-derive the document the plan describes and compare it to the text that
    # was asked for. The policy checked its own arithmetic; this trusts none of
    # it, and it is policy-agnostic — a policy with no notion of regions is
    # validated by exactly the same check.
    _validate_plan_reassembles(plan, stored_chunks, new_text)

    return {
        "staged": staged,
        "document": document,
        "data_item": old_data,
        "stored_chunks": stored_chunks,
        "plan": plan,
    }


def _validate_plan_reassembles(plan: ChunkPlan, stored_chunks: List[dict], new_text: str) -> None:
    """Refuse a plan whose chunks do not reassemble into exactly the new text.

    Also catches what a region-level check cannot: duplicate or missing final
    positions, which would silently reorder or drop a chunk.
    """
    stored_text_by_id = {str(node["id"]): node["text"] for node in stored_chunks}
    placed: dict = {}
    for chunk in plan.fresh:
        placed[chunk.chunk_index] = chunk.text
    for chunk_id, index in list(plan.reused.items()) + list(plan.kept_moves.items()):
        placed[index] = stored_text_by_id[chunk_id]
    # Kept chunks the policy did not move keep their stored position.
    moved = set(plan.reused) | set(plan.kept_moves)
    deleted = set(plan.deleted_ids)
    for node in stored_chunks:
        chunk_id = str(node["id"])
        if chunk_id in moved or chunk_id in deleted:
            continue
        placed.setdefault(int(node.get("chunk_index", -1)), node["text"])

    expected_positions = set(range(len(placed)))
    if set(placed) != expected_positions:
        raise IncrementalUpdateNotPossible(
            "plan leaves chunk positions with gaps or duplicates",
            RefusalReason.CHUNKS_NOT_TILING,
        )
    if "".join(placed[index] for index in sorted(placed)) != new_text:
        raise IncrementalUpdateNotPossible(
            "incremental plan would lose or corrupt content",
            RefusalReason.CHUNKS_NOT_TILING,
        )


async def _write_and_publish(
    bundle: dict,
    data_id: UUID,
    dataset,
    user: User,
    node_set: Optional[List[str]],
    graph_model: type[BaseModel],
    custom_prompt: Optional[str],
    pipeline_run_id: UUID,
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
    plan: ChunkPlan = bundle["plan"]
    stored_by_id = {str(node["id"]): node for node in stored_chunks}

    # The run that is recording this write, not a fresh id: the provenance
    # stamped on every node and edge below embeds it, and that stamp is what
    # makes the write rollbackable and auditable by run.
    context = PipelineContext(
        user=user,
        data_item=bundle.get("data_item"),
        dataset=dataset,
        pipeline_run_id=pipeline_run_id,
        pipeline_name=PIPELINE_NAME,
    )

    # Surviving chunks are rebuilt HERE, not in the policy: the rehydration
    # carries every stored field across, and a field it drops is erased rather
    # than reset (adapters replace a node's whole property set on MERGE). The
    # plan names them; the writer knows how to rebuild them.
    reused_chunks = [
        _rehydrate_chunk(document, stored_by_id[chunk_id], index)
        for chunk_id, index in plan.reused.items()
    ]

    cognify_config = get_cognify_config()
    if plan.fresh:
        # Same extraction + summarization the cognify pipeline runs, with the
        # same ontology resolution, model, and prompt plumbing.
        summaries = await extract_graph_and_summarize(
            plan.fresh,
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
    if plan.deleted_ids:
        doomed = await delete_chunks_incremental(plan.deleted_ids, dataset_id, data_id)
        await _prune_ledger_rows(data_id, dataset_id, doomed)

    # -- Renumber kept chunks whose position shifted --------------------------- #
    shifted_chunks = [
        _rehydrate_chunk(document, stored_by_id[chunk_id], index)
        for chunk_id, index in plan.kept_moves.items()
    ]
    if shifted_chunks:
        await _restore_repositioned_chunks(shifted_chunks, context)

    # -- PUBLISH: content + metadata + token count + stamp, atomically --------- #
    replaced = set(plan.deleted_ids) | set(plan.reused)
    surviving_tokens = sum(
        int(node.get("chunk_size", 0)) for node in stored_chunks if str(node["id"]) not in replaced
    )
    new_tokens = sum(chunk.chunk_size for chunk in plan.fresh)
    new_tokens += sum(int(stored_by_id[chunk_id].get("chunk_size", 0)) for chunk_id in plan.reused)
    await _publish_staged(data_id, dataset_id, staged, surviving_tokens + new_tokens, node_set)

    added_chunks = len(plan.fresh) + len(plan.reused)
    kept_count = len(stored_chunks) - len(replaced)
    logger.info(
        "incremental update: %d regions, kept %d chunks, deleted %d, added %d "
        "(%d reused), reindexed %d",
        plan.regions,
        kept_count,
        len(plan.deleted_ids),
        added_chunks,
        len(reused_chunks),
        len(shifted_chunks),
    )
    return {
        "status": "incremental",
        "regions": plan.regions,
        "deleted_chunks": len(plan.deleted_ids),
        "added_chunks": added_chunks,
        "reused_chunks": len(reused_chunks),
        "kept_chunks": kept_count,
        "reindexed_chunks": len(shifted_chunks),
    }
