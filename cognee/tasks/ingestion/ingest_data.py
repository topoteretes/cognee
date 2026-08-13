import json
import inspect
from uuid import UUID, uuid4
from typing import Union, BinaryIO, Any, List, Optional

import cognee.modules.ingestion as ingestion
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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
    resolve_data_id,
    get_dataset_data,
    load_or_create_datasets,
)

from .save_data_item_to_storage import save_data_item_to_storage
from .data_item_to_text_file import data_item_to_text_file
from .data_item import DataItem


async def ingest_data(
    data: Any,
    dataset_name: str,
    user: User,
    node_set: Optional[List[str]] = None,
    dataset_id: UUID = None,
    preferred_loaders: dict[str, dict[str, Any]] = None,
    importance_weight: float = 0.5,
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

        db_engine = get_relational_engine()

        # Pre-loop: resolve or mint data_id for every item and cache intermediate
        # results to avoid repeating expensive I/O in the main loop. Dedup is a
        # dataset-scoped LOOKUP (identify); a miss mints a random id — two
        # identical items in one batch share the first mint.
        data_point_ids = []
        precomputed_items = {}
        batch_id_by_hash: dict = {}
        for data_item in data:
            underlying_data = data_item.data if isinstance(data_item, DataItem) else data_item
            item_data_id = data_item.data_id if isinstance(data_item, DataItem) else None

            original_file_path = await save_data_item_to_storage(underlying_data)
            actual_file_path = get_data_file_path(original_file_path)

            async with open_data_file(actual_file_path) as file:
                classified_data = ingestion.classify(file)
                item_content_hash = classified_data.get_identifier()
                data_id = await ingestion.identify(classified_data, user, dataset.id)

            if item_data_id is not None:
                # A pinned id may be one the user held before a fork/update —
                # resolve it (exact, then legacy) instead of minting a new row
                # under a legacy value. Unknown pins stay as-is (dlt mints
                # stable ids through this path deliberately).
                resolved_pin = await resolve_data_id(dataset.id, item_data_id)
                data_id = resolved_pin if resolved_pin is not None else item_data_id
            elif data_id is None:
                data_id = batch_id_by_hash.get(item_content_hash) or uuid4()
            batch_id_by_hash.setdefault(item_content_hash, data_id)

            data_point_ids.append(data_id)
            precomputed_items[id(data_item)] = {
                "original_file_path": original_file_path,
                "actual_file_path": actual_file_path,
                "data_id": data_id,
            }

        existing_data_map: dict = {}
        if data_point_ids:
            async with db_engine.get_async_session() as session:
                result = await session.execute(select(Data).filter(Data.id.in_(data_point_ids)))
                for dp in result.scalars().all():
                    existing_data_map[str(dp.id)] = dp

        for data_item in data:
            # Support for DataItem (custom label + data + optional data_id / external_metadata)
            current_label = None
            underlying_data = data_item
            item_data_id = None
            item_external_metadata = None

            if isinstance(data_item, DataItem):
                underlying_data = data_item.data
                current_label = data_item.label
                item_data_id = data_item.data_id
                item_external_metadata = data_item.external_metadata

            # Retrieve cached intermediate results from pre-loop to avoid re-processing
            cached = precomputed_items.get(id(data_item), {})
            original_file_path = cached.get("original_file_path")
            actual_file_path = cached.get("actual_file_path")

            # Store all input data as text files in Cognee data storage
            cognee_storage_file_path, loader_engine = await data_item_to_text_file(
                actual_file_path,
                preferred_loaders,
            )

            if loader_engine is None:
                raise IngestionError("Loader cannot be None")

            # Use data_id computed in pre-loop
            data_id = cached.get("data_id")

            # Find metadata from original file
            # Standard flow: extract metadata from both original and stored files
            async with open_data_file(original_file_path) as file:
                classified_data = ingestion.classify(file)
                original_file_metadata = classified_data.get_metadata()

            # Find metadata from Cognee data storage text file
            async with open_data_file(cognee_storage_file_path) as file:
                classified_data = ingestion.classify(file)
                storage_file_metadata = classified_data.get_metadata()

            data_point = existing_data_map.get(str(data_id))

            # TODO: Maybe allow getting of external metadata through ingestion loader?
            ext_metadata = get_external_metadata_dict(data_item)

            # Merge DataItem.external_metadata if present
            if item_external_metadata:
                ext_metadata.update(item_external_metadata)

            if node_set:
                ext_metadata["node_set"] = node_set

            if data_point is not None:
                # Content-change detection: reset pipeline_status when content changed
                new_content_hash = original_file_metadata["content_hash"]
                content_changed = str(data_point.content_hash) != str(new_content_hash)

                # Rows are dataset-scoped (the startup migration backfills
                # legacy rows). A row of another dataset can only reach this
                # branch through a mispinned data_id — never mutate it.
                if str(data_point.dataset_id) != str(dataset.id):
                    raise IngestionError(
                        f"Data {data_point.id} belongs to dataset {data_point.dataset_id}; "
                        f"refusing to touch it from dataset {dataset.id}."
                    )

                data_point.name = original_file_metadata["name"]
                data_point.raw_data_location = cognee_storage_file_path
                data_point.original_data_location = original_file_metadata["file_path"]
                data_point.extension = storage_file_metadata["extension"]
                data_point.mime_type = storage_file_metadata["mime_type"]
                data_point.original_extension = original_file_metadata["extension"]
                data_point.original_mime_type = original_file_metadata["mime_type"]
                data_point.loader_engine = loader_engine.loader_name
                data_point.owner_id = user.id
                data_point.content_hash = new_content_hash
                data_point.raw_content_hash = storage_file_metadata["content_hash"]
                data_point.data_size = original_file_metadata["file_size"]
                data_point.external_metadata = ext_metadata
                data_point.node_set = json.dumps(node_set) if node_set else None
                data_point.tenant_id = user.tenant_id if user.tenant_id else None
                # Absent means "leave unchanged": a re-ingest without a label
                # (current_label None) must not clear a previously stored one.
                if current_label is not None:
                    data_point.label = current_label

                if content_changed:
                    data_point.pipeline_status = {}

                existing_data_points.append(data_point)
                dataset_data_map[str(data_point.id)] = True
            else:
                if str(data_id) in dataset_data_map:
                    continue

                data_point = Data(
                    id=data_id,
                    dataset_id=dataset.id,
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
                    label=current_label,
                    importance_weight=importance_weight,
                )

                new_datapoints.append(data_point)
                dataset_data_map[str(data_point.id)] = True

        async with db_engine.get_async_session() as session:
            for data_point in existing_data_points:
                await session.merge(data_point)
            session.add_all(new_datapoints)
            await session.commit()

        return existing_data_points + new_datapoints

    # Concurrent ingests PINNED to the same data_id (dlt derives stable ids;
    # update() re-ingests under the document's own id) can both try to INSERT
    # the same primary key — the loser hits "UNIQUE constraint failed: data.id";
    # retrying re-reads the committed row and takes the existing-data branch.
    # Unpinned rows mint random ids and dedup by lookup instead: a concurrent
    # same-content race there can produce two rows, which under document
    # semantics are simply two documents. File writes are content-addressed,
    # so re-running is idempotent.
    try:
        return await store_data_to_dataset(
            data, dataset_name, user, node_set, dataset_id, preferred_loaders
        )
    except IntegrityError:
        return await store_data_to_dataset(
            data, dataset_name, user, node_set, dataset_id, preferred_loaders
        )
