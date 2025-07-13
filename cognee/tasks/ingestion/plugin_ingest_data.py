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
from cognee.api.v1.add.config import get_s3_config


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
        user_datasets = await get_specific_user_permission_datasets(user.id, ["write"])
        existing_datasets = await get_authorized_existing_datasets(user.id, dataset_name, ["write"])

        datasets = await load_or_create_datasets(
            user_datasets, existing_datasets, dataset_name, user, dataset_id
        )

        dataset = datasets[0]

        new_datapoints = []
        existing_data_points = []
        dataset_new_data_points = []

        # Get existing dataset data for deduplication (preserve existing logic)
        dataset_data: list[Data] = await get_dataset_data(dataset.id)
        dataset_data_map = {str(data.id): True for data in dataset_data}

        for data_item in data:
            file_path = await save_data_item_to_storage(data_item, dataset_name)

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
                    classified_data = ingestion.classify(file, s3fs=fs)

            # Preserve all existing data processing logic
            data_id = ingestion.identify(classified_data, user)
            file_metadata = classified_data.get_metadata()

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
                data_point.name = file_metadata["name"]
                data_point.raw_data_location = file_metadata["file_path"]
                data_point.extension = file_metadata["extension"]
                data_point.mime_type = file_metadata["mime_type"]
                data_point.owner_id = user.id
                data_point.content_hash = file_metadata["content_hash"]
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
                    name=file_metadata["name"],
                    raw_data_location=file_metadata["file_path"],
                    extension=file_metadata["extension"],
                    mime_type=file_metadata["mime_type"],
                    owner_id=user.id,
                    content_hash=file_metadata["content_hash"],
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
