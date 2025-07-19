import json
import inspect
from uuid import UUID
from typing import Union, BinaryIO, Any, List, Optional

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    get_dataset_data,
    load_or_create_datasets,
)
from cognee.shared.logging_utils import get_logger

from .save_data_item_to_storage import save_data_item_to_storage
from .adapters import LoaderToIngestionAdapter
from cognee.infrastructure.files.storage.s3_config import get_s3_config


logger = get_logger(__name__)


async def plugin_ingest_data(
    data: Any,
    dataset_name: str,
    user: User,
    node_set: Optional[List[str]] = None,
    dataset_id: UUID = None,
    preferred_loaders: Optional[List[str]] = None,
    loader_config: Optional[dict] = None,
):
    """
    Plugin-based data ingestion using the loader system.

    This function maintains full backward compatibility with the existing
    ingest_data function while adding support for the new loader system.

    Args:
        data: The data to ingest
        dataset_name: Name of the dataset
        user: User object for permissions
        node_set: Optional node set for organization
        dataset_id: Optional specific dataset ID
        preferred_loaders: List of preferred loader names to try first
        loader_config: Configuration for specific loaders

    Returns:
        List of Data objects that were ingested
    """
    if not user:
        user = await get_default_user()

    # Ensure NLTK data is downloaded (preserves automatic download behavior)
    def ensure_nltk_data():
        """Download required NLTK data if not already present."""
        try:
            import nltk

            # Download essential NLTK data used by the system
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("maxent_ne_chunker", quiet=True)
            nltk.download("words", quiet=True)
            logger.info("NLTK data verified/downloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")

    # Download NLTK data once per session
    if not hasattr(plugin_ingest_data, "_nltk_initialized"):
        ensure_nltk_data()
        plugin_ingest_data._nltk_initialized = True

    # Initialize S3 support (maintain existing behavior)
    s3_config = get_s3_config()
    fs = None
    if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
        import s3fs

        fs = s3fs.S3FileSystem(
            key=s3_config.aws_access_key_id, secret=s3_config.aws_secret_access_key, anon=False
        )

    # Initialize the loader adapter
    loader_adapter = LoaderToIngestionAdapter()

    def open_data_file(file_path: str):
        """Open file with S3 support (preserves existing behavior)."""
        if file_path.startswith("s3://"):
            return fs.open(file_path, mode="rb")
        else:
            local_path = file_path.replace("file://", "")
            return open(local_path, mode="rb")

    def get_external_metadata_dict(data_item: Union[BinaryIO, str, Any]) -> dict[str, Any]:
        """Get external metadata (preserves existing behavior)."""
        if hasattr(data_item, "dict") and inspect.ismethod(getattr(data_item, "dict")):
            return {"metadata": data_item.dict(), "origin": str(type(data_item))}
        else:
            return {}

    async def store_data_to_dataset(
        data: Any,
        dataset_name: str,
        user: User,
        node_set: Optional[List[str]] = None,
        dataset_id: UUID = None,
    ):
        """
        Core data storage logic with plugin-based file processing.

        This function preserves all existing permission and database logic
        while using the new loader system for file processing.
        """
        logger.info(f"Plugin-based ingestion starting for dataset: {dataset_name}")

        # Preserve existing dataset creation and permission logic
        if dataset_id:
            # Retrieve existing dataset by ID
            dataset = await get_specific_user_permission_datasets(user.id, "write", [dataset_id])
            # Convert from list to Dataset element
            if isinstance(dataset, list):
                dataset = dataset[0]
        else:
            # Find existing dataset or create a new one by name
            existing_datasets = await get_authorized_existing_datasets(
                datasets=[dataset_name], permission_type="write", user=user
            )
            datasets = await load_or_create_datasets(
                dataset_names=[dataset_name], existing_datasets=existing_datasets, user=user
            )
            if isinstance(datasets, list):
                dataset = datasets[0]

        new_datapoints = []
        existing_data_points = []
        dataset_new_data_points = []

        # Get existing dataset data for deduplication (preserve existing logic)
        dataset_data: list[Data] = await get_dataset_data(dataset.id)
        dataset_data_map = {str(data.id): True for data in dataset_data}

        for data_item in data:
            file_path = await save_data_item_to_storage(data_item)

            # NEW: Use loader system or existing classification based on data type
            try:
                if loader_adapter.is_text_content(data_item):
                    # Handle text content (preserve existing behavior)
                    logger.info("Processing text content with existing system")
                    classified_data = ingestion.classify(data_item)
                else:
                    # Use loader system for file paths
                    logger.info(f"Processing file with loader system: {file_path}")
                    classified_data = await loader_adapter.process_file_with_loaders(
                        file_path,
                        s3fs=fs,
                        preferred_loaders=preferred_loaders,
                        loader_config=loader_config,
                    )

            except Exception as e:
                logger.warning(f"Plugin system failed for {file_path}, falling back: {e}")
                # Fallback to existing system for full backward compatibility
                with open_data_file(file_path) as file:
                    classified_data = ingestion.classify(file)

            # Preserve all existing data processing logic
            data_id = ingestion.identify(classified_data, user)
            file_metadata = classified_data.get_metadata()

            # Ensure metadata has all required fields with fallbacks
            def get_metadata_field(metadata, field_name, default_value=""):
                """Get metadata field with fallback handling."""
                if field_name in metadata and metadata[field_name] is not None:
                    return metadata[field_name]

                logger.warning(f"Missing metadata field '{field_name}', using fallback")

                # Provide fallbacks based on available information
                if field_name == "name":
                    if "file_path" in metadata and metadata["file_path"]:
                        import os

                        return os.path.basename(str(metadata["file_path"])).split(".")[0]
                    elif file_path:
                        import os

                        return os.path.basename(str(file_path)).split(".")[0]
                    else:
                        content_hash = metadata.get("content_hash", str(data_id))[:8]
                        return f"content_{content_hash}"
                elif field_name == "file_path":
                    # Use the actual file path returned by save_data_item_to_storage
                    return file_path
                elif field_name == "extension":
                    if "file_path" in metadata and metadata["file_path"]:
                        import os

                        _, ext = os.path.splitext(str(metadata["file_path"]))
                        return ext.lstrip(".") if ext else "txt"
                    elif file_path:
                        import os

                        _, ext = os.path.splitext(str(file_path))
                        return ext.lstrip(".") if ext else "txt"
                    return "txt"
                elif field_name == "mime_type":
                    ext = get_metadata_field(metadata, "extension", "txt")
                    mime_map = {
                        "txt": "text/plain",
                        "md": "text/markdown",
                        "pdf": "application/pdf",
                        "json": "application/json",
                        "csv": "text/csv",
                    }
                    return mime_map.get(ext.lower(), "text/plain")
                elif field_name == "content_hash":
                    # Extract the raw content hash for compatibility with deletion system
                    content_identifier = classified_data.get_identifier()
                    # Remove content type prefix if present (e.g., "text_abc123" -> "abc123")
                    if "_" in content_identifier:
                        return content_identifier.split("_", 1)[-1]
                    return content_identifier

                return default_value

            from sqlalchemy import select

            db_engine = get_relational_engine()

            # Check if data should be updated (preserve existing logic)
            async with db_engine.get_async_session() as session:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()

            ext_metadata = get_external_metadata_dict(data_item)

            if node_set:
                ext_metadata["node_set"] = node_set

            # Preserve existing data point creation/update logic
            if data_point is not None:
                data_point.name = get_metadata_field(file_metadata, "name")
                data_point.raw_data_location = get_metadata_field(file_metadata, "file_path")
                data_point.extension = get_metadata_field(file_metadata, "extension")
                data_point.mime_type = get_metadata_field(file_metadata, "mime_type")
                data_point.owner_id = user.id
                data_point.content_hash = get_metadata_field(file_metadata, "content_hash")
                data_point.external_metadata = ext_metadata
                data_point.node_set = json.dumps(node_set) if node_set else None

                if str(data_point.id) in dataset_data_map:
                    existing_data_points.append(data_point)
                else:
                    dataset_new_data_points.append(data_point)
                    dataset_data_map[str(data_point.id)] = True
            else:
                if str(data_id) in dataset_data_map:
                    continue

                data_point = Data(
                    id=data_id,
                    name=get_metadata_field(file_metadata, "name"),
                    raw_data_location=get_metadata_field(file_metadata, "file_path"),
                    extension=get_metadata_field(file_metadata, "extension"),
                    mime_type=get_metadata_field(file_metadata, "mime_type"),
                    owner_id=user.id,
                    content_hash=get_metadata_field(file_metadata, "content_hash"),
                    external_metadata=ext_metadata,
                    node_set=json.dumps(node_set) if node_set else None,
                    token_count=-1,
                )

                new_datapoints.append(data_point)
                dataset_data_map[str(data_point.id)] = True

        # Preserve existing database operations
        async with db_engine.get_async_session() as session:
            if dataset not in session:
                session.add(dataset)

            if len(new_datapoints) > 0:
                dataset.data.extend(new_datapoints)

            if len(existing_data_points) > 0:
                for data_point in existing_data_points:
                    await session.merge(data_point)

            if len(dataset_new_data_points) > 0:
                dataset.data.extend(dataset_new_data_points)

            await session.merge(dataset)
            await session.commit()

        logger.info(
            f"Plugin-based ingestion completed. New: {len(new_datapoints)}, "
            + f"Updated: {len(existing_data_points)}, Dataset new: {len(dataset_new_data_points)}"
        )

        return existing_data_points + dataset_new_data_points + new_datapoints

    return await store_data_to_dataset(data, dataset_name, user, node_set, dataset_id)
