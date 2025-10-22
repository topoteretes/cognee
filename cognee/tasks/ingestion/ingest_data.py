import json
import inspect
from uuid import UUID
from typing import Union, BinaryIO, Any, List, Optional

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.modules.data.methods import (
    get_authorized_existing_datasets,
    get_dataset_data,
    load_or_create_datasets,
)

from .save_data_item_to_storage import save_data_item_to_storage
from .data_item_to_text_file import data_item_to_text_file


async def ingest_data(
    data: Any,
    dataset_name: str,
    user: User,
    node_set: Optional[List[str]] = None,
    dataset_id: UUID = None,
    preferred_loaders: dict[str, dict[str, Any]] = None,
):
    if not user:
        user = await get_default_user()

    def get_external_metadata_dict(data_item: Union[BinaryIO, str, Any]) -> dict[str, Any]:
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
        preferred_loaders: dict[str, dict[str, Any]] = None,
    ):
        new_datapoints = []
        existing_data_points = []
        dataset_new_data_points = []

        if not isinstance(data, list):
            # Convert data to a list as we work with lists further down.
            data = [data]

        if dataset_id:
            # Retrieve existing dataset
            dataset = await get_specific_user_permission_datasets(user.id, "write", [dataset_id])
            # Convert from list to Dataset element
            if isinstance(dataset, list):
                dataset = dataset[0]
        else:
            # Find existing dataset or create a new one
            existing_datasets = await get_authorized_existing_datasets(
                user=user, permission_type="write", datasets=[dataset_name]
            )
            dataset = await load_or_create_datasets(
                dataset_names=[dataset_name],
                existing_datasets=existing_datasets,
                user=user,
            )
            if isinstance(dataset, list):
                dataset = dataset[0]

        dataset_data: list[Data] = await get_dataset_data(dataset.id)
        dataset_data_map = {str(data.id): True for data in dataset_data}

        for data_item in data:
            # Get file path of data item or create a file if it doesn't exist
            original_file_path = await save_data_item_to_storage(data_item)
            # Transform file path to be OS usable
            actual_file_path = get_data_file_path(original_file_path)

            # Store all input data as text files in Cognee data storage
            cognee_storage_file_path, loader_engine = await data_item_to_text_file(
                actual_file_path,
                preferred_loaders,
            )

            if loader_engine is None:
                raise IngestionError("Loader cannot be None")

            # Find metadata from original file
            # Standard flow: extract metadata from both original and stored files
            async with open_data_file(original_file_path) as file:
                classified_data = ingestion.classify(file)

                # data_id is the hash of original file contents + owner id to avoid duplicate data

                data_id = ingestion.identify(classified_data, user)
                original_file_metadata = classified_data.get_metadata()

            # Find metadata from Cognee data storage text file
            async with open_data_file(cognee_storage_file_path) as file:
                classified_data = ingestion.classify(file)
                storage_file_metadata = classified_data.get_metadata()

            from sqlalchemy import select

            db_engine = get_relational_engine()

            # Check to see if data should be updated
            async with db_engine.get_async_session() as session:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()

            # TODO: Maybe allow getting of external metadata through ingestion loader?
            ext_metadata = get_external_metadata_dict(data_item)

            if node_set:
                ext_metadata["node_set"] = node_set

            if data_point is not None:
                data_point.name = original_file_metadata["name"]
                data_point.raw_data_location = cognee_storage_file_path
                data_point.original_data_location = original_file_metadata["file_path"]
                data_point.extension = storage_file_metadata["extension"]
                data_point.mime_type = storage_file_metadata["mime_type"]
                data_point.original_extension = original_file_metadata["extension"]
                data_point.original_mime_type = original_file_metadata["mime_type"]
                data_point.loader_engine = loader_engine.loader_name
                data_point.owner_id = user.id
                data_point.content_hash = original_file_metadata["content_hash"]
                data_point.raw_content_hash = storage_file_metadata["content_hash"]
                data_point.file_size = original_file_metadata["file_size"]
                data_point.external_metadata = ext_metadata
                data_point.node_set = json.dumps(node_set) if node_set else None
                data_point.tenant_id = user.tenant_id if user.tenant_id else None

                # Check if data is already in dataset
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
                    name=original_file_metadata["name"],
                    raw_data_location=cognee_storage_file_path,
                    original_data_location=original_file_metadata["file_path"],
                    extension=storage_file_metadata["extension"],
                    mime_type=storage_file_metadata["mime_type"],
                    original_extension=original_file_metadata["extension"],
                    original_mime_type=original_file_metadata["mime_type"],
                    loader_engine=loader_engine.loader_name,
                    owner_id=user.id,
                    content_hash=original_file_metadata["content_hash"],
                    raw_content_hash=storage_file_metadata["content_hash"],
                    external_metadata=ext_metadata,
                    node_set=json.dumps(node_set) if node_set else None,
                    data_size=original_file_metadata["file_size"],
                    tenant_id=user.tenant_id if user.tenant_id else None,
                    pipeline_status={},
                    token_count=-1,
                )

                new_datapoints.append(data_point)
                dataset_data_map[str(data_point.id)] = True

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

        return existing_data_points + dataset_new_data_points + new_datapoints

    return await store_data_to_dataset(
        data, dataset_name, user, node_set, dataset_id, preferred_loaders
    )
