from typing import List, Union, BinaryIO
from os import path
import asyncio
import dlt
import duckdb
import cognee.modules.ingestion as ingestion
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.files.storage import LocalStorage
from cognee.modules.discovery import discover_directory_datasets
from cognee.utils import send_telemetry
from cognee.base_config import get_base_config
base_config = get_base_config()
from cognee.infrastructure.databases.relational.config import get_relationaldb_config

relational_config = get_relationaldb_config()


async def add(data: Union[BinaryIO, List[BinaryIO], str, List[str]], dataset_name: str = None):
    if isinstance(data, str):
        # data is a data directory path
        if "data://" in data:
            return await add_data_directory(data.replace("data://", ""), dataset_name)
        # data is a file path
        if "file://" in data:
            return await add([data], dataset_name)
        # data is a text
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
        return await add_files(file_paths, dataset_name)

    return []

async def add_files(file_paths: List[str], dataset_name: str):
    # infra_config = infrastructure_config.get_config()
    data_directory_path = base_config.data_root_directory

    LocalStorage.ensure_directory_exists(relational_config.database_directory_path)

    processed_file_paths = []

    for file_path in file_paths:
        file_path = file_path.replace("file://", "")

        if data_directory_path not in file_path:
            file_name = file_path.split("/")[-1]
            file_directory_path = data_directory_path + "/" + (dataset_name.replace(".", "/") + "/" if dataset_name != "root" else "")
            dataset_file_path = path.join(file_directory_path, file_name)

            LocalStorage.ensure_directory_exists(file_directory_path)

            LocalStorage.copy_file(file_path, dataset_file_path)
            processed_file_paths.append(dataset_file_path)
        else:
            processed_file_paths.append(file_path)

    db = duckdb.connect(relational_config.db_file_path)

    destination = dlt.destinations.duckdb(
        credentials = db,
    )

    pipeline = dlt.pipeline(
        pipeline_name = "file_load_from_filesystem",
        destination = destination,
    )

    @dlt.resource(standalone = True, merge_key = "id")
    def data_resources(file_paths: str):
        for file_path in file_paths:
            with open(file_path.replace("file://", ""), mode = "rb") as file:
                classified_data = ingestion.classify(file)

                data_id = ingestion.identify(classified_data)

                file_metadata = classified_data.get_metadata()

                yield {
                    "id": data_id,
                    "name": file_metadata["name"],
                    "file_path": file_metadata["file_path"],
                    "extension": file_metadata["extension"],
                    "mime_type": file_metadata["mime_type"],
                    "keywords": "|".join(file_metadata["keywords"]),
                }

    run_info = pipeline.run(
        data_resources(processed_file_paths),
        table_name = "file_metadata",
        dataset_name = dataset_name.replace(" ", "_").replace(".", "_") if dataset_name is not None else "main_dataset",
        write_disposition = "merge",
    )
    send_telemetry("cognee.add")

    return run_info

async def add_data_directory(data_path: str, dataset_name: str = None):
    datasets = discover_directory_datasets(data_path)

    results = []

    for key in datasets.keys():
        if dataset_name is None or key.startswith(dataset_name):
            results.append(add(datasets[key], dataset_name = key))

    return await asyncio.gather(*results)

def save_data_to_file(data: Union[str, BinaryIO], dataset_name: str, filename: str = None):
    data_directory_path = base_config.data_root_directory

    classified_data = ingestion.classify(data, filename)
    # data_id = ingestion.identify(classified_data)

    storage_path = data_directory_path + "/" + dataset_name.replace(".", "/")
    LocalStorage.ensure_directory_exists(storage_path)

    file_metadata = classified_data.get_metadata()
    file_name = file_metadata["name"]
    LocalStorage(storage_path).store(file_name, classified_data.get_data())

    return "file://" + storage_path + "/" + file_name
