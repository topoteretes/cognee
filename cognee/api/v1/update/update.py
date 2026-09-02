from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any, Dict

from cognee.modules.pipelines.models import PipelineRunInfo
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.api.v1.add import add
from cognee.api.v1.cognify import cognify
from cognee.api.v1.datasets import datasets


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
) -> Union[Dict[str, PipelineRunInfo], List[PipelineRunInfo]]:
    """
    Update existing data in Cognee.

    The document keeps its ``data_id`` across updates: the replacement is
    re-ingested pinned to the resolved id (exact, or the recorded pre-fork
    ``legacy_id``), so externally held id mappings never break. Exactly one
    document is replaced per call — lists of more than one item are rejected.
    An id that resolves to no document raises ``UpdateTargetNotFoundError``
    (404) — update() never creates documents; use add() for that.

    DLT source manifests get a row-level update instead: when the resolved id
    names a DLT manifest (system_metadata.source == "dlt_source"), the new
    source is re-ingested and diffed against the stored manifest by
    content-addressed row ids. Removed rows are deleted from the graph/vector
    stores, new and edited rows are processed, unchanged rows are left
    untouched (no re-embedding), and a later cognify() skips the manifest.
    The new source must keep the same source name (the manifest identity); it
    returns a summary dict (rows_added / rows_edited / rows_removed /
    rows_unchanged / fk_edges_repaired) instead of PipelineRunInfo.

    Supported Input Types:
        - **Text strings**: Direct text content (str) - any string not starting with "/" or "file://"
        - **File paths**: Local file paths as strings in these formats:
            * Absolute paths: "/path/to/document.pdf"
            * File URLs: "file:///path/to/document.pdf" or "file://relative/path.txt"
            * S3 paths: "s3://bucket-name/path/to/file.pdf"
        - **Binary file objects**: File handles/streams (BinaryIO)
        - **Lists**: Multiple files or text strings in a single call

    Supported File Formats:
        - Text files (.txt, .md, .csv)
        - PDFs (.pdf)
        - Images (.png, .jpg, .jpeg) - extracted via OCR/vision models
        - Audio files (.mp3, .wav) - transcribed to text
        - Code files (.py, .js, .ts, etc.) - parsed for structure and content
        - Office documents (.docx, .pptx)

            Workflow:
        1. **Data Resolution**: Resolves file paths and validates accessibility
        2. **Content Extraction**: Extracts text content from various file formats
        3. **Dataset Storage**: Stores processed content in the specified dataset
        4. **Metadata Tracking**: Records file metadata, timestamps, and user permissions
        5. **Permission Assignment**: Grants user read/write/delete/share permissions on dataset

    Args:
        data_id: UUID of existing data to update
        data: The latest version of the data. Can be:
            - Single text string: "Your text content here"
            - Absolute file path: "/path/to/document.pdf"
            - File URL: "file:///absolute/path/to/document.pdf" or "file://relative/path.txt"
            - S3 path: "s3://my-bucket/documents/file.pdf"
            - List of mixed types: ["text content", "/path/file.pdf", "file://doc.txt", file_handle]
            - Binary file object: open("file.txt", "rb")
        dataset_id: UUID of the dataset holding the document (required).
        user: User object for authentication and permissions. Uses default user if None.
              Default user: "default_user@example.com" (created automatically on first use).
              Users can only access datasets they have permissions for.
        node_set: Optional list of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        vector_db_config: Optional configuration for vector database (for custom setups).
        graph_db_config: Optional configuration for graph database (for custom setups).

    Returns:
        PipelineRunInfo: Information about the ingestion pipeline execution including:
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
    from cognee.tasks.ingestion.dlt_utils import is_dlt_source_manifest

    if isinstance(data, list):
        if len(data) != 1:
            raise IngestionError(
                f"update() replaces exactly one document; got a list of {len(data)} items."
            )
        data = data[0]

    # The document KEEPS its data_id through updates. Resolve the incoming id
    # (exact, then pre-fork legacy_id) and re-ingest pinned to the resolved
    # id. An id that resolves to nothing is a caller error, not a create:
    # ids are random uuid4s now, so a stale or mistyped id can never match —
    # silently creating a second document would hide the mistake as
    # duplication. add() is the path for new documents.
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

    # DLT source manifests get a row-level update: the resolved id names the
    # manifest, the new source is diffed against it by content-addressed row
    # ids, and only the delta is processed — removed rows deleted, added rows
    # embedded and wired, unchanged rows untouched. Everything else keeps the
    # delete + pinned re-add + re-cognify path below. The check runs on the
    # dataset-resolved row (not a context-free lookup) so forked ids keep the
    # resolution contract above.
    if old_row is not None and is_dlt_source_manifest(old_row):
        from cognee.modules.data.methods.get_dataset import get_dataset
        from cognee.tasks.ingestion.update_dlt_source import update_dlt_source_rows

        dataset = await get_dataset(user.id, dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found for user {user.id}.")

        return await update_dlt_source_rows(
            data_record=old_row,
            data=data,
            dataset=dataset,
            user=user,
        )

    await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
    )

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
        # update() returns the run info for the caller to inspect — an errored
        # run is a valid return value here, not an exception.
        raise_on_error=False,
    )

    return cognify_run
