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
from cognee.api.v1.update.incremental import IncrementalUpdateNotPossible, incremental_update
from cognee.shared.logging_utils import get_logger

logger = get_logger("update")


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
        graph_db_config: Optional configuration for graph database (for custom setups).
        chunk_level_diff: When True (default), diff the new content against the stored
                 processed text and replace only the chunks the edit touched — unaffected
                 chunks keep their nodes, entities, and summaries. Falls back to the full
                 delete + pinned re-add + cognify flow when chunk-level preconditions are
                 not met (first ingestion, non-text content, unverified graph adapter).
                 Permission errors always propagate and never trigger the fallback.

    Returns:
        With chunk_level_diff, a summary dict:
            {"status": "incremental" | "unchanged", "deleted_chunks": n, "added_chunks": n,
             "reused_chunks": n, "kept_chunks": n, "reindexed_chunks": n}
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
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        old_row = await session.get(Data, resolved_id)
        if old_row is not None:
            preserved_legacy_id = old_row.legacy_id

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
            )
        except IncrementalUpdateNotPossible as reason:
            logger.info("chunk-level update not possible (%s); running full update", reason)

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

    # Restore fork lineage onto the recreated row: a fork document's pre-fork
    # id must keep resolving across updates.
    if preserved_legacy_id is not None:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            new_row = await session.get(Data, pinned_id)
            if new_row is not None and new_row.legacy_id is None:
                new_row.legacy_id = preserved_legacy_id
                await session.commit()

    cognify_run = await cognify(
        datasets=[dataset_id],
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
        graph_model=graph_model,
        custom_prompt=custom_prompt,
    )

    return cognify_run
