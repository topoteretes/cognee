"""Chunk-level incremental update: stage, plan, validate, write, publish.

Flow (only runs from the update endpoint):

1. Enforce dataset-level "write" AND "delete" permission plus data
   membership/ownership. Delete too, because this path destroys graph state —
   the full fallback reaches the same destruction through delete_data, which
   requires it, so the two branches must not disagree.
2. Read the OLD processed text from ``Data.raw_data_location`` and the stored
   chunk nodes from the graph, ordered by their position in that text.
3. STAGE the new content: the input runs through the same loader machinery as
   ingestion (content-addressed storage write) — but the ``Data`` row is NOT
   touched. Readers keep resolving the coherent old version.
4. PLAN, through a swappable ``ChunkPolicy`` (default: diff into disjoint
   changed regions, re-chunk each at the budget its replaced chunks recorded,
   keep everything between them untouched). The policy returns a COMPLETE
   decision — fresh chunks, surviving ids and their new positions, dead ids —
   so this module executes rather than finishes planning. Its result is then
   re-derived here and compared to the new text byte-for-byte: the policy
   checked its own arithmetic, and the orchestrator trusts none of it. Only
   now — staging and validation done — is a pipeline run record created;
   refused updates leave no run-record noise.
5. Write: extract ONLY the fresh chunks, in bounded batches, through the
   standard graph-extraction and storage tasks (attributed to the same
   ``data_id``), retire replaced chunk ownership through the shared deletion
   planner, and renumber moved survivors.
6. PUBLISH in one relational transaction: content location, hashes, size,
   token count, and the processed stamp flip together. A crash anywhere
   before the publish leaves the row on the old content; the stored chunks
   then no longer tile the stored text, so the next touch fails closed into a
   full rebuild (self-heal).

Raises IncrementalUpdateNotPossible when preconditions fail, carrying a typed
``RefusalReason`` (unsupported backend, chunker, or metadata; no baseline;
chunks not tiling; unreadable text) — the caller decides to run the full
update instead, and can tell a permanent misconfiguration from a first
ingestion in the logs. Permission errors are NOT refusals: they propagate.
"""

import json
from enum import Enum
from pathlib import PureWindowsPath
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.cognify.config import get_cognify_config
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
from cognee.tasks.graph.detect_contradictions import detect_contradictions
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
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
from cognee.modules.data.methods import (
    StagedContent,
    get_authorized_dataset,
    get_data,
    is_data_processed,
    mark_data_processed,
    merged_external_metadata,
    publish_updated_data,
)
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
from cognee.modules.ingestion import classify, save_data_to_file
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points

logger = get_logger("incremental_update")

PIPELINE_NAME = "cognify_pipeline"  # attribute to the cognify pipeline's status key
# Distinct run-record name: dashboards see incremental runs without touching
# the skip logic that keys on cognify_pipeline's per-item status.
RUN_PIPELINE_NAME = "incremental_update_pipeline"
# Matches cognify's own fallback when chunks_per_batch is unset.
DEFAULT_CHUNKS_PER_BATCH = 2000


class RefusalReason(str, Enum):
    """Why a chunk-level update fell back to the full flow.

    Every refusal used to surface as one free-text message and one log line, so
    a permanent misconfiguration (an incompatible chunker, an unsupported
    backend) looked exactly like a first ingestion. The reason is logged as a
    structured field so they are separable.
    """

    UNSUPPORTED_BACKEND = "unsupported_backend"
    UNSUPPORTED_CHUNKER = "unsupported_chunker"
    UNSUPPORTED_METADATA = "unsupported_metadata"
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


async def _get_stored_chunks(document_id: UUID) -> List[dict]:
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


async def _require_chunk_scoped_ownership(
    stored_chunks: List[dict], dataset_id: UUID, data_id: UUID
) -> None:
    """Refuse baselines whose chunks predate v2 ownership.

    The graph provenance marker also exists on older graphs whose artifacts
    carry only one document-scoped v1 ref. Those graphs are safe for whole-
    document deletion, but not for deleting one chunk by its v2 ref.
    """
    graph_engine = await get_graph_engine()
    chunk_ids = [str(node["id"]) for node in stored_chunks]
    delete_data = await graph_engine.get_node_delete_data(chunk_ids)
    for chunk_id in chunk_ids:
        expected_ref = make_chunk_source_ref_key(dataset_id, data_id, UUID(chunk_id))
        snapshot = delete_data.get(chunk_id)
        if snapshot is None or expected_ref not in snapshot.source_ref_keys:
            raise IncrementalUpdateNotPossible(
                f"stored chunk {chunk_id} has no chunk-scoped ownership baseline",
                RefusalReason.NO_BASELINE,
            )


async def recorded_chunk_budget(data_id: UUID, dataset_id: UUID, user: User) -> Optional[int]:
    """The token budget the document's stored chunks were cut against, if usable.

    The full-flow fallback re-cognifies the document; without this it would
    cut at the current default, and a document cognified at a custom
    ``chunk_size`` would come back at a different granularity — and, where the
    default exceeds the incremental path's limit, be refused by the budget
    guard on every later update. Returns None when the graph holds no usable
    baseline or the recorded budget is larger than the current provider limit
    (then the default is the only safe cut).

    Resolves the dataset's databases as the DATASET OWNER, like
    ``incremental_update`` and ``run_tasks``. Passing the caller would send a
    collaborator's lookup to a different per-user store, find no chunks there,
    and silently report "no recorded budget" — reintroducing the granularity
    drift this helper exists to prevent, for exactly the collaborator the
    dataset ACL was meant to serve.

    Never raises: this is an optimization on the fallback path, so a lookup it
    cannot complete degrades to the current default rather than failing the
    update.
    """
    try:
        dataset = await get_authorized_dataset(user, dataset_id, "write")
        async with set_database_global_context_variables(dataset_id, dataset.owner_id):
            stored_chunks = await _get_stored_chunks(data_id)
    except (IncrementalUpdateNotPossible, PermissionDeniedError):
        return None
    recorded = next(
        (int(node["max_chunk_tokens"]) for node in stored_chunks if node.get("max_chunk_tokens")),
        None,
    )
    if recorded is None or recorded > await get_max_chunk_tokens():
        return None
    return recorded


def _require_stored_chunks_tile(stored_chunks: List[dict], old_text: str) -> None:
    """Refuse extra, missing, or wrongly ordered chunks before any shortcut."""
    if "".join(node["text"] for node in stored_chunks) != old_text:
        raise IncrementalUpdateNotPossible(
            "stored chunks do not tile the stored document text",
            RefusalReason.CHUNKS_NOT_TILING,
        )


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
        external_metadata=json.dumps(merged_external_metadata(data, node_set), indent=4),
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


def _misindexed_chunks(document: Document, stored_chunks: List[dict]) -> List[DocumentChunk]:
    """Stored chunks whose recorded index disagrees with their actual position.

    Drift the planner never intended: a crash between a delete and its
    renumbering leaves survivors carrying stale indexes. Rehydrated with their
    EXISTING id and the corrected index, so re-storing them upserts the graph
    node and refreshes the vector payload rather than creating anything.

    Planned moves are a different thing and come from ``plan.kept_moves``; this
    only repairs a document whose stored order is already inconsistent.
    """
    return [
        _rehydrate_chunk(document, node, position)
        for position, node in enumerate(stored_chunks)
        if int(node.get("chunk_index", -1)) != position
    ]


async def _restore_repositioned_chunks(chunks: List[DocumentChunk], _context) -> None:
    """Write back chunks whose ONLY change is their position (kept or reused).

    Incremental preflight requires both narrow operations. The graph adapter
    patches ONLY the index, and the vector adapter changes only its payload.
    The stored node remains the source of truth for every other field.
    """
    vector_engine = await get_vector_engine_async()
    graph_engine = await get_graph_engine()
    await graph_engine.update_chunk_index({str(chunk.id): chunk.chunk_index for chunk in chunks})
    await vector_engine.update_payload(
        "DocumentChunk_text",
        {str(chunk.id): {"chunk_index": chunk.chunk_index} for chunk in chunks},
    )


async def _stage_new_content(data, preferred_loaders) -> StagedContent:
    """Run the input through ingestion's loader machinery without row writes.

    Mirrors ingest_data's loader and metadata derivation. Upload bytes use a
    content-addressed staging name so a same-name replacement cannot overwrite
    the current row's original file before publish.
    """
    source_data = data.data if isinstance(data, DataItem) else data
    upload_metadata = None
    if hasattr(source_data, "file") and getattr(source_data, "filename", None):
        # Normal ingestion stores uploads by their user-visible filename with
        # overwrite=True. That is unsafe for staging: updating report.pdf could
        # replace the CURRENT row's original file before publish. Keep the
        # display name separately and stage bytes under a content-addressed path.
        upload_metadata = classify(source_data.file, filename=source_data.filename).get_metadata()
        extension = upload_metadata["extension"]
        suffix = f".{extension}" if extension else ""
        staged_filename = f"staged_original_{upload_metadata['content_hash']}{suffix}"
        original_file_path = await save_data_to_file(
            source_data.file,
            filename=staged_filename,
        )
    else:
        original_file_path = await save_data_item_to_storage(data)
    actual_file_path = get_data_file_path(original_file_path)

    storage_file_path, loader_engine = await data_item_to_text_file(
        actual_file_path, preferred_loaders
    )
    if loader_engine is None:
        raise IncrementalUpdateNotPossible(
            "no loader accepted the new content", RefusalReason.UNREADABLE_TEXT
        )
    if isinstance(storage_file_path, LoaderResult):
        # A loader may hand back a LoaderResult (its own identity and route
        # stamp, as dlt does) instead of a plain path; only the stored text's
        # path matters here — ingest_data unwraps it the same way.
        storage_file_path = storage_file_path.file_path

    async with open_data_file(original_file_path) as file:
        original_metadata = classify(file).get_metadata()
    if upload_metadata is not None:
        original_metadata["name"] = PureWindowsPath(str(source_data.filename)).stem
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


def _changed_staged_metadata(data, old_data: Data, staged: StagedContent) -> list[str]:
    """Return metadata changes that need document-wide full-update handling."""
    fields = [
        "extension",
        "mime_type",
        "original_extension",
        "original_mime_type",
        "loader_engine",
    ]
    # Direct text gets an internal content-derived filename, so its name is
    # expected to change with its text. User-named uploads and streams are not.
    source_data = data.data if isinstance(data, DataItem) else data
    if hasattr(source_data, "filename") or hasattr(source_data, "name"):
        fields.append("name")
    return [
        field for field in fields if getattr(old_data, field, None) != getattr(staged, field, None)
    ]


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
    await mark_data_processed(data_id, dataset.id)
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
    # Dataset membership above is the access rule, not Data.owner_id: a
    # collaborator holding the dataset's write+delete ACL may update a row the
    # dataset owner ingested. datasets.delete_data authorizes exactly that way,
    # and the full fallback reaches this same row through it — checking row
    # ownership here would deny on the fast branch what the slow branch allows.
    old_data = await get_data(user.id, data_id, dataset.id, verify_owner=False)
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
        # Resolve the dataset's databases as the DATASET OWNER, matching
        # run_tasks and datasets.delete_data. Passing the caller would send a
        # collaborator's update to a different per-user store than the one
        # cognify and delete use for this dataset.
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            graph_engine = await get_graph_engine()
            vector_engine = await get_vector_engine_async()
            if not getattr(graph_engine, "supports_incremental_chunk_updates", False):
                raise IncrementalUpdateNotPossible(
                    f"graph backend {type(graph_engine).__name__} does not support "
                    "chunk-level updates",
                    RefusalReason.UNSUPPORTED_BACKEND,
                )
            if not getattr(vector_engine, "supports_payload_update", False):
                raise IncrementalUpdateNotPossible(
                    f"vector backend {type(vector_engine).__name__} does not support "
                    "payload-only chunk moves",
                    RefusalReason.UNSUPPORTED_BACKEND,
                )
            if not await stores_provenance_in_graph(graph_engine):
                raise IncrementalUpdateNotPossible(
                    "the selected graph does not store ownership provenance in-graph",
                    RefusalReason.UNSUPPORTED_BACKEND,
                )
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
    old_data = await get_data(user.id, data_id, dataset_id, verify_owner=False)
    if old_data is None or not old_data.raw_data_location:
        raise IncrementalUpdateNotPossible(
            "data row disappeared before the update ran", RefusalReason.NO_BASELINE
        )

    old_text = await _read_processed_text(old_data.raw_data_location)
    stored_chunks = await _get_stored_chunks(data_id)
    await _require_chunk_scoped_ownership(stored_chunks, dataset_id, data_id)
    _require_stored_chunks_tile(stored_chunks, old_text)

    # Stage the new content through ingestion's own loader machinery. The Data
    # row is NOT written: readers keep resolving the old version until publish.
    staged = await _stage_new_content(data, preferred_loaders)
    new_text = await _read_processed_text(staged.raw_data_location)
    content_unchanged = staged.content_hash == old_data.content_hash and new_text == old_text

    changed_metadata = _changed_staged_metadata(data, old_data, staged)
    if changed_metadata:
        raise IncrementalUpdateNotPossible(
            f"replacement metadata changed ({', '.join(changed_metadata)})",
            RefusalReason.UNSUPPORTED_METADATA,
        )

    document = _build_document(old_data, staged, node_set)

    if content_unchanged:
        # Same content re-submitted. Two things can still be stale, and both
        # are reported rather than repaired here: chunk indexes left behind by
        # a crash between an earlier delete and its renumbering, and a missing
        # cognify-completion stamp (without which a later cognify would redo
        # the whole document). A missing stamp counts as a repair so the
        # self-heal keeps its reach — it is a write either way, and every write
        # belongs under a run.
        shifted = _misindexed_chunks(document, stored_chunks)
        needs_stamp = not await is_data_processed(data_id, dataset_id)
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
    moved = set(plan.reused) | set(plan.kept_moves)
    deleted = set(plan.deleted_ids)

    # (final position, text) for every chunk the document will have. Collected
    # as a LIST, not a dict: two chunks claiming one position must be visible
    # as a duplicate rather than one silently overwriting the other.
    placed = [(chunk.chunk_index, chunk.text) for chunk in plan.fresh]
    placed += [
        (index, stored_text_by_id[chunk_id])
        for chunk_id, index in list(plan.reused.items()) + list(plan.kept_moves.items())
    ]
    # Kept chunks the policy did not move stay at their stored position.
    placed += [
        (int(node.get("chunk_index", -1)), node["text"])
        for node in stored_chunks
        if str(node["id"]) not in moved and str(node["id"]) not in deleted
    ]

    positions = [index for index, _ in placed]
    if sorted(positions) != list(range(len(placed))):
        raise IncrementalUpdateNotPossible(
            "plan leaves chunk positions with gaps or duplicates",
            RefusalReason.CHUNKS_NOT_TILING,
        )
    if "".join(text for _, text in sorted(placed)) != new_text:
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
    # Same extraction + summarization the cognify pipeline runs, with the same
    # ontology resolution, model, and prompt plumbing — and the same batch
    # bound. Cognify gets its batching from the pipeline task machinery
    # (task_config={"batch_size": ...}), which this path does not run through,
    # so the slicing is explicit here. Unbounded, a rewrite of most of a large
    # document becomes one oversized extraction step with no intermediate
    # progress and a single all-or-nothing failure.
    batch_size = cognify_config.chunks_per_batch or DEFAULT_CHUNKS_PER_BATCH
    for start in range(0, len(plan.fresh), batch_size):
        batch = plan.fresh[start : start + batch_size]
        summaries = await extract_graph_and_summarize(
            batch,
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
        await delete_chunks_incremental(plan.deleted_ids, dataset_id, data_id)

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
    await publish_updated_data(data_id, dataset_id, staged, surviving_tokens + new_tokens, node_set)

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
