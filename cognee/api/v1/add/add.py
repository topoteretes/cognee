from typing import List, Union, BinaryIO
from os import path
import asyncio
import dlt
import duckdb

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.files.storage import LocalStorage
from cognee.modules.ingestion import get_matched_datasets, save_data_to_file
from cognee.shared.utils import send_telemetry
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.relational import get_relational_config, get_relational_engine, create_db_and_tables
from cognee.modules.users.methods import create_default_user, get_default_user
from cognee.modules.users.permissions.methods import give_permission_on_document
from cognee.modules.users.models import User
from cognee.modules.data.operations.ensure_dataset_exists import ensure_dataset_exists

async def add(data: Union[BinaryIO, List[BinaryIO], str, List[str]], dataset_name: str = "main_dataset", user: User = None):
    await create_db_and_tables()

    if isinstance(data, str):
        if "data://" in data:
            # data is a data directory path
            datasets = get_matched_datasets(data.replace("data://", ""), dataset_name)
            return await asyncio.gather(*[add(file_paths, dataset_name) for [dataset_name, file_paths] in datasets])

        if "file://" in data:
            # data is a file path
            return await add([data], dataset_name)

        # data is text
        else:
            file_path = save_data_to_file(data, dataset_name)
            return await add([file_path], dataset_name)

    if hasattr(data, "file"):
        file_path = save_data_to_file(data.file, dataset_name, filename = data.filename)
        return await add([file_path], dataset_name)

    # data is a list of file paths or texts
    file_paths = []

    for data_item in data:
        if hasattr(data_item, "file"):
            file_paths.append(save_data_to_file(data_item, dataset_name, filename = data_item.filename))
        elif isinstance(data_item, str) and (
            data_item.startswith("/") or data_item.startswith("file://")
        ):
            file_paths.append(data_item)
        elif isinstance(data_item, str):
            file_paths.append(save_data_to_file(data_item, dataset_name))

    if len(file_paths) > 0:
        return await add_files(file_paths, dataset_name, user)

    return []

async def add_files(file_paths: List[str], dataset_name: str,  user):
    base_config = get_base_config()
    data_directory_path = base_config.data_root_directory

    processed_file_paths = []

    for file_path in file_paths:
        file_path = file_path.replace("file://", "")

        if data_directory_path not in file_path:
            file_name = file_path.split("/")[-1]
            file_directory_path = data_directory_path + "/" + (dataset_name.replace(".", "/") + "/" if dataset_name != "main_dataset" else "")
            dataset_file_path = path.join(file_directory_path, file_name)

            LocalStorage.ensure_directory_exists(file_directory_path)

            LocalStorage.copy_file(file_path, dataset_file_path)
            processed_file_paths.append(dataset_file_path)
        else:
            processed_file_paths.append(file_path)

    relational_config = get_relational_config()

    if relational_config.db_provider == "duckdb":
        db = duckdb.connect(relational_config.db_file_path)

        destination = dlt.destinations.duckdb(
          credentials = db,
        )
    else:
        destination = dlt.destinations.postgres(
            credentials = {
                "host": relational_config.db_host,
                "port": relational_config.db_port,
                "user": relational_config.db_user,
                "password": relational_config.db_password,
                "database": relational_config.db_name,
            },
        )

    pipeline = dlt.pipeline(
        pipeline_name = "file_load_from_filesystem",
        destination = destination,
    )

    dataset_name = dataset_name.replace(" ", "_").replace(".", "_") if dataset_name is not None else "main_dataset"
    dataset = await ensure_dataset_exists(dataset_name)

    @dlt.resource(standalone = True, merge_key = "id")
    async def data_resources(file_paths: str, user: User):
        for file_path in file_paths:
            with open(file_path.replace("file://", ""), mode = "rb") as file:
                classified_data = ingestion.classify(file)

                data_id = ingestion.identify(classified_data)

                file_metadata = classified_data.get_metadata()

                from sqlalchemy import select
                from cognee.modules.data.models import Data
                db_engine = get_relational_engine()
                async with db_engine.get_async_session() as session:
                    data = (await session.execute(
                        select(Data).filter(Data.id == data_id)
                    )).scalar_one_or_none()

                    if data is not None:
                        data.name = file_metadata["name"]
                        data.raw_data_location = file_metadata["file_path"]
                        data.extension = file_metadata["extension"]
                        data.mime_type = file_metadata["mime_type"]

                        await session.merge(data)
                        await session.commit()
                    else:
                        data = Data(
                            id = data_id,
                            name = file_metadata["name"],
                            raw_data_location = file_metadata["file_path"],
                            extension = file_metadata["extension"],
                            mime_type = file_metadata["mime_type"],
                        )
                        dataset.data.append(data)

                        await session.merge(dataset)

                        await session.commit()

                yield {
                    "id": data_id,
                    "name": file_metadata["name"],
                    "file_path": file_metadata["file_path"],
                    "extension": file_metadata["extension"],
                    "mime_type": file_metadata["mime_type"],
                }

                await give_permission_on_document(user, data_id, "read")
                await give_permission_on_document(user, data_id, "write")


    if user is None:
        user = await get_default_user()

        if user is None:
            user = await create_default_user()

    run_info = pipeline.run(
        data_resources(processed_file_paths, user),
        table_name = "file_metadata",
        dataset_name = dataset_name,
        write_disposition = "merge",
    )
    send_telemetry("cognee.add")

    return run_info
