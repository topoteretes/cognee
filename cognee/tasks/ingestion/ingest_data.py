import dlt
import s3fs
import json
import inspect
from typing import Union, BinaryIO, Any, List, Optional
import cognee.modules.ingestion as ingestion
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.methods import create_dataset, get_dataset_data, get_datasets_by_name
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.models.DatasetData import DatasetData
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import give_permission_on_document
from .get_dlt_destination import get_dlt_destination
from .save_data_item_to_storage import save_data_item_to_storage


from cognee.api.v1.add.config import get_s3_config


async def ingest_data(
    data: Any, dataset_name: str, user: User, node_set: Optional[List[str]] = None
):
    destination = get_dlt_destination()

    if not user:
        user = await get_default_user()

    pipeline = dlt.pipeline(
        pipeline_name="metadata_extraction_pipeline",
        destination=destination,
    )

    s3_config = get_s3_config()

    fs = None
    if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
        fs = s3fs.S3FileSystem(
            key=s3_config.aws_access_key_id, secret=s3_config.aws_secret_access_key, anon=False
        )

    def open_data_file(file_path: str):
        if file_path.startswith("s3://"):
            return fs.open(file_path, mode="rb")
        else:
            local_path = file_path.replace("file://", "")
            return open(local_path, mode="rb")

    def get_external_metadata_dict(data_item: Union[BinaryIO, str, Any]) -> dict[str, Any]:
        if hasattr(data_item, "dict") and inspect.ismethod(getattr(data_item, "dict")):
            return {"metadata": data_item.dict(), "origin": str(type(data_item))}
        else:
            return {}

    @dlt.resource(standalone=True, primary_key="id", merge_key="id")
    async def data_resources(file_paths: List[str], user: User):
        for file_path in file_paths:
            with open_data_file(file_path) as file:
                if file_path.startswith("s3://"):
                    classified_data = ingestion.classify(file, s3fs=fs)
                else:
                    classified_data = ingestion.classify(file)
                data_id = ingestion.identify(classified_data, user)
                file_metadata = classified_data.get_metadata()
                yield {
                    "id": data_id,
                    "name": file_metadata["name"],
                    "file_path": file_metadata["file_path"],
                    "extension": file_metadata["extension"],
                    "mime_type": file_metadata["mime_type"],
                    "content_hash": file_metadata["content_hash"],
                    "owner_id": str(user.id),
                    "node_set": json.dumps(node_set) if node_set else None,
                }

    async def store_data_to_dataset(
        data: Any, dataset_name: str, user: User, node_set: Optional[List[str]] = None
    ):
        if not isinstance(data, list):
            # Convert data to a list as we work with lists further down.
            data = [data]

        file_paths = []

        # Process data
        for data_item in data:
            file_path = await save_data_item_to_storage(data_item, dataset_name)

            file_paths.append(file_path)

            # Ingest data and add metadata
            # with open(file_path.replace("file://", ""), mode="rb") as file:
            with open_data_file(file_path) as file:
                classified_data = ingestion.classify(file, s3fs=fs)

                # data_id is the hash of file contents + owner id to avoid duplicate data
                data_id = ingestion.identify(classified_data, user)

                file_metadata = classified_data.get_metadata()

                from sqlalchemy import select

                from cognee.modules.data.models import Data

                db_engine = get_relational_engine()

                async with db_engine.get_async_session() as session:
                    dataset = await create_dataset(dataset_name, user, session)

                    # Check to see if data should be updated
                    data_point = (
                        await session.execute(select(Data).filter(Data.id == data_id))
                    ).scalar_one_or_none()

                    ext_metadata = get_external_metadata_dict(data_item)
                    if node_set:
                        ext_metadata["node_set"] = node_set

                    if data_point is not None:
                        data_point.name = file_metadata["name"]
                        data_point.raw_data_location = file_metadata["file_path"]
                        data_point.extension = file_metadata["extension"]
                        data_point.mime_type = file_metadata["mime_type"]
                        data_point.owner_id = user.id
                        data_point.content_hash = file_metadata["content_hash"]
                        data_point.external_metadata = ext_metadata
                        data_point.node_set = json.dumps(node_set) if node_set else None
                        await session.merge(data_point)
                    else:
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

                    # Check if data is already in dataset
                    dataset_data = (
                        await session.execute(
                            select(DatasetData).filter(
                                DatasetData.data_id == data_id, DatasetData.dataset_id == dataset.id
                            )
                        )
                    ).scalar_one_or_none()
                    # If data is not present in dataset add it
                    if dataset_data is None:
                        dataset.data.append(data_point)

                    await session.commit()

                await give_permission_on_document(user, data_id, "read")
                await give_permission_on_document(user, data_id, "write")

        return file_paths

    db_engine = get_relational_engine()

    file_paths = await store_data_to_dataset(data, dataset_name, user, node_set)

    # Note: DLT pipeline has its own event loop, therefore objects created in another event loop
    # can't be used inside the pipeline
    if db_engine.engine.dialect.name == "sqlite":
        # To use sqlite with dlt dataset_name must be set to "main".
        # Sqlite doesn't support schemas
        pipeline.run(
            data_resources(file_paths, user),
            table_name="file_metadata",
            dataset_name="main",
            write_disposition="merge",
        )
    else:
        # Data should be stored in the same schema to allow deduplication
        pipeline.run(
            data_resources(file_paths, user),
            table_name="file_metadata",
            dataset_name="public",
            write_disposition="merge",
        )

    datasets = await get_datasets_by_name(dataset_name, user.id)

    # In case no files were processed no dataset will be created
    if datasets:
        dataset = datasets[0]
        data_documents = await get_dataset_data(dataset_id=dataset.id)
        return data_documents
    return []
