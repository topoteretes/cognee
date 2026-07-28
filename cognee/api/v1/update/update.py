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
        chunk_level_diff: When True (default), diff the new content against the stored
                 processed text and replace only the chunks the edit touched — unaffected
                 chunks keep their nodes, entities, and summaries. Falls back to the full
                 delete + re-add + cognify flow when chunk-level preconditions are not met
                 (first ingestion, non-text content, unverified graph adapter). Permission
                 errors always propagate and never trigger the fallback.

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

    if chunk_level_diff:
        # Chunk-level incremental path: diff the new text against the stored
        # processed text, replace only the affected chunks. Falls through to
        # the full delete+re-add flow when its preconditions aren't met
        # (first ingestion, non-text content, stored chunks unavailable).
        # Permission errors propagate — they must never trigger the fallback.
        try:
            return await incremental_update(
                data_id=data_id,
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
        data_id=data_id,
        user=user,
    )

    await add(
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
