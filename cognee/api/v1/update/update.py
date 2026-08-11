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
        dataset_name: Name of the dataset to store data in. Defaults to "main_dataset".
                    Create separate datasets to organize different knowledge domains.
        user: User object for authentication and permissions. Uses default user if None.
              Default user: "default_user@example.com" (created automatically on first use).
              Users can only access datasets they have permissions for.
        node_set: Optional list of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        vector_db_config: Optional configuration for vector database (for custom setups).
        graph_db_config: Optional configuration for graph database (for custom setups).
        dataset_id: Optional specific dataset UUID to use instead of dataset_name.

    Returns:
        PipelineRunInfo: Information about the ingestion pipeline execution including:
            - Pipeline run ID for tracking
            - Dataset ID where data was stored
            - Processing status and any errors
            - Execution timestamps and metadata
    """
    if not user:
        user = await get_default_user()

    # Lineage capture (flatten-on-write): the replacement row must keep
    # resolving under every id the user ever held for this document.
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods import resolve_data_id
    from cognee.modules.data.models import Data

    original_external_id = None
    resolved_id = await resolve_data_id(dataset_id, data_id)
    if resolved_id is not None:
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            old_row = await session.get(Data, resolved_id)
            if old_row is not None:
                original_external_id = old_row.legacy_id or old_row.id

    await datasets.delete_data(
        dataset_id=dataset_id,
        data_id=data_id,
        user=user,
    )

    add_run = await add(
        data=data,
        dataset_id=dataset_id,
        user=user,
        node_set=node_set,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        preferred_loaders=preferred_loaders,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
    )

    # Stamp the replacement row with the original external id, so the user's
    # pre-update mapping keeps resolving. First-wins: never overwrite an
    # existing lineage (the row may itself be a fork of an older id).
    ingestion_info = getattr(add_run, "data_ingestion_info", None) or []
    new_data_id = ingestion_info[0].get("data_id") if ingestion_info else None
    if (
        original_external_id is not None
        and new_data_id is not None
        and str(new_data_id) != str(original_external_id)
    ):
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            new_row = await session.get(Data, new_data_id)
            if new_row is not None and new_row.legacy_id is None:
                new_row.legacy_id = original_external_id
                await session.merge(new_row)
                await session.commit()

    cognify_run = await cognify(
        datasets=[dataset_id],
        user=user,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
        data_cache=data_cache,
    )

    return cognify_run
