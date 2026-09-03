from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any, Dict

from pydantic import BaseModel

from cognee.modules.pipelines.models import PipelineRunInfo
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify
from cognee.api.v1.datasets import datasets
from cognee.api.v1.update.incremental import (
    IncrementalUpdateNotPossible,
    incremental_update,
    recorded_chunk_budget,
)
from cognee.modules.chunking.chunk_policy import DEFAULT_CHUNK_POLICY, ChunkPolicy
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.shared.logging_utils import get_logger

logger = get_logger("update")


async def _restore_row_lineage(
    data_id: UUID, legacy_id: Optional[UUID], owner_id: Optional[UUID]
) -> None:
    """Carry the replaced row's identity onto the one re-ingestion just minted.

    The full fallback deletes the row and re-adds it, and ``add()`` builds a
    fresh ``Data`` with ``owner_id=user.id``. Both values must survive that:

    - ``legacy_id`` so a fork document's pre-fork id keeps resolving;
    - ``owner_id`` so a collaborator updating a document does not silently
      become its owner. Update is authorized by the dataset ACL, which says
      nothing about who owns the row.
    """
    if legacy_id is None and owner_id is None:
        return

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await session.get(Data, data_id)
        if row is None:
            return
        if legacy_id is not None and row.legacy_id is None:
            row.legacy_id = legacy_id
        if owner_id is not None:
            row.owner_id = owner_id
        await session.commit()


async def update(
    data_id: UUID,
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_id: UUID,
    user: User = None,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    preferred_loaders: dict[str, dict[str, Any]] = None,
    incremental_loading: bool = True,
    data_cache: bool = True,
    chunk_level_diff: bool = True,
    graph_model: type[BaseModel] = KnowledgeGraph,
    custom_prompt: Optional[str] = None,
    chunker: type = TextChunker,
    policy: ChunkPolicy = DEFAULT_CHUNK_POLICY,
) -> Union[Dict[str, PipelineRunInfo], List[PipelineRunInfo], dict]:
    """
    Update existing data in Cognee.

    The document keeps its ``data_id`` across updates — on EVERY path. The
    incoming id is resolved first (exact, or the recorded pre-fork
    ``legacy_id``), the chunk-level incremental path operates on the resolved
    row in place, and the full-rebuild fallback re-ingests pinned to the same
    id — so externally held id mappings never break, incremental or not.
    Exactly one document is replaced per call — lists of more than one item
    are rejected. An id that resolves to no document raises
    ``UpdateTargetNotFoundError`` (404) — update() never creates documents;
    use add() for that.

    Supported Input Types:
        - **Text strings**: Direct text content (str) - any string not starting with "/" or "file://"
        - **File paths**: Local file paths as strings in these formats:
            * Absolute paths: "/path/to/document.pdf"
            * File URLs: "file:///path/to/document.pdf" or "file://relative/path.txt"
            * S3 paths: "s3://bucket-name/path/to/file.pdf"
        - **Binary file objects**: File handles/streams (BinaryIO)

    Supported File Formats:
        - Text files (.txt, .md, .csv)
        - PDFs (.pdf)
        - Images (.png, .jpg, .jpeg) - extracted via OCR/vision models
        - Audio files (.mp3, .wav) - transcribed to text
        - Code files (.py, .js, .ts, etc.) - parsed for structure and content
        - Office documents (.docx, .pptx)

    Args:
        data_id: UUID of existing data to update (current or pre-fork)
        data: The latest version of the data. Can be:
            - Single text string: "Your text content here"
            - Absolute file path: "/path/to/document.pdf"
            - File URL: "file:///absolute/path/to/document.pdf" or "file://relative/path.txt"
            - S3 path: "s3://my-bucket/documents/file.pdf"
            - Binary file object: open("file.txt", "rb")
        dataset_id: UUID of the dataset holding the document (required).
        user: User object for authentication and permissions. Uses default user if None.
              Default user: "default_user@example.com" (created automatically on first use).
              Users can only access datasets they have permissions for.
        node_set: Optional list of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        vector_db_config: Optional configuration for vector database (for custom setups).
                 Chunk-level incremental updates do not support per-call config
                 forwarding: when provided, the update runs full ingestion
                 instead (a warning is logged). For incremental updates,
                 configure stores through environment settings and the
                 dataset-context database routing system.
        graph_db_config: Optional configuration for graph database (for custom setups).
                 Same routing note as vector_db_config.
        chunk_level_diff: When True (default), diff the new content against the stored
                 processed text and replace only the chunks the edit touched — unaffected
                 chunks keep their nodes, entities, and summaries. Falls back to the full
                 delete + pinned re-add + cognify flow when chunk-level preconditions are
                 not met (first ingestion, non-text content, unverified graph adapter).
                 Permission errors always propagate and never trigger the fallback.
        chunker: Chunking strategy. Must match the one that built the document's stored
                 chunks — a mismatch is refused (and falls back) rather than surfacing
                 as a tiling failure. Chunk-level path only.
        policy: Decides which chunks exist after the edit and what happens to the old
                 ones. Replaceable without touching storage or update orchestration.
                 Chunk-level path only; not exposed on the HTTP route.

    Returns:
        With chunk_level_diff, a summary dict with the same keys for either status:
            {"status": "incremental" | "unchanged", "regions": n, "deleted_chunks": n,
             "added_chunks": n, "reused_chunks": n, "kept_chunks": n, "reindexed_chunks": n}
        Otherwise PipelineRunInfo: Information about the ingestion pipeline execution including:
            - Pipeline run ID for tracking
            - Dataset ID where data was stored
            - Processing status and any errors
            - Execution timestamps and metadata
    """
    if not user:
        user = await get_default_user()

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods import resolve_data_id
    from cognee.modules.data.models import Data
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.tasks.ingestion.data_item import DataItem

    if isinstance(data, list):
        if len(data) != 1:
            raise IngestionError(
                f"update() replaces exactly one document; got a list of {len(data)} items."
            )
        data = data[0]

    # The document KEEPS its data_id through updates. Resolve the incoming id
    # (exact, then pre-fork legacy_id) once, up front: the incremental path
    # operates on the resolved row, and the fallback re-ingests pinned to it.
    # An id that resolves to nothing is a caller error, not a create: ids are
    # random uuid4s now, so a stale or mistyped id can never match — silently
    # creating a second document would hide the mistake as duplication.
    # add() is the path for new documents.
    resolved_id = await resolve_data_id(dataset_id, data_id)
    if resolved_id is None:
        from cognee.api.v1.exceptions import UpdateTargetNotFoundError

        raise UpdateTargetNotFoundError(data_id=data_id, dataset_id=dataset_id)
    pinned_id = resolved_id

    preserved_legacy_id = None
    preserved_owner_id = None
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        old_row = await session.get(Data, resolved_id)
        if old_row is not None:
            preserved_legacy_id = old_row.legacy_id
            preserved_owner_id = old_row.owner_id

    data_item_changes_metadata = isinstance(data, DataItem) and (
        data.label is not None or data.external_metadata is not None
    )
    if chunk_level_diff and (node_set or data_item_changes_metadata):
        # Warning, not info: the response carries no hint that the slower,
        # costlier full rebuild ran instead of the chunk-level update.
        logger.warning(
            "Chunk-level incremental update does not reconcile document metadata or "
            "node_set; running full update"
        )
        chunk_level_diff = False

    if chunk_level_diff and (graph_model is not KnowledgeGraph or custom_prompt is not None):
        # The baseline does not persist the model/prompt that produced its
        # graph. Applying a new configuration only to fresh chunks would mix
        # extraction schemas or rules inside one document.
        logger.warning(
            "Chunk-level incremental update supports only the default graph model and prompt; "
            "running full update"
        )
        chunk_level_diff = False

    if chunk_level_diff and (vector_db_config is not None or graph_db_config is not None):
        # The chunk-level incremental engine resolves its stores through the
        # dataset-context routing system, not per-call config dicts — running
        # it with these params would silently read and write the DEFAULT
        # stores. The full ingestion flow honors them, so it runs instead.
        logger.warning(
            "Chunk-level incremental update is not supported with per-call "
            "vector_db_config/graph_db_config forwarding; running full "
            "ingestion instead. To get incremental updates, configure your "
            "stores through environment settings and the dataset-context "
            "database routing system instead of per-call config dicts."
        )
        chunk_level_diff = False

    # The fallback re-cognifies the whole document. It keeps the chunk budget
    # the stored chunks record so the document's granularity survives the
    # rebuild; None (no baseline, or a recorded budget the current provider
    # cannot take) means the current default.
    fallback_chunk_size = None

    if chunk_level_diff:
        # Chunk-level incremental path: diff the new text against the stored
        # processed text, replace only the affected chunks — the Data row is
        # updated in place, so the id trivially survives. Falls through to
        # the full flow when its preconditions aren't met (first ingestion,
        # non-text content, stored chunks unavailable). Permission errors
        # propagate — they must never trigger the fallback.
        try:
            return await incremental_update(
                data_id=pinned_id,
                data=data,
                dataset_id=dataset_id,
                user=user,
                node_set=node_set,
                preferred_loaders=preferred_loaders,
                graph_model=graph_model,
                custom_prompt=custom_prompt,
                chunker=chunker,
                policy=policy,
            )
        except IncrementalUpdateNotPossible as refusal:
            # The reason is a structured field, not just prose: an unsupported
            # chunker and a first ingestion produce the same sentence otherwise,
            # so a permanent misconfiguration is indistinguishable from a
            # one-off in the logs. A warning, not info: the fallback re-extracts
            # the whole document and the caller's response carries no hint of
            # it (pipeline-run info instead of the incremental summary), so
            # this line is the only place the downgrade and its cause show up.
            logger.warning(
                "chunk-level update not possible (%s); running full update",
                refusal,
                extra={"refusal_reason": refusal.reason.value},
            )
            fallback_chunk_size = await recorded_chunk_budget(pinned_id, dataset_id, user)

    await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=pinned_id,
        user=user,
    )

    # Fallback keeps the id too: re-ingest pinned to the resolved id instead
    # of minting a fresh one (the fallback-churn defect the reconciliation
    # review identified — callers must never lose their handle to a fallback).
    if isinstance(data, DataItem):
        data.data_id = pinned_id
        pinned_item = data
    else:
        pinned_item = DataItem(data=data, data_id=pinned_id)

    await add(
        data=pinned_item,
        dataset_id=dataset_id,
        user=user,
        node_set=node_set,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        preferred_loaders=preferred_loaders,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
    )

    await _restore_row_lineage(pinned_id, preserved_legacy_id, preserved_owner_id)

    cognify_run = await cognify(
        datasets=[dataset_id],
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
        graph_model=graph_model,
        custom_prompt=custom_prompt,
        chunk_size=fallback_chunk_size,
        # update() returns the run info for the caller to inspect — an errored
        # run is a valid return value here, not an exception.
        raise_on_error=False,
    )

    return cognify_run
