from typing import List, Union
from os import path, listdir
import asyncio
import dlt
import duckdb
from cognee.root_dir import get_absolute_path
import cognee.modules.ingestion as ingestion
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.files import get_file_metadata
from cognee.infrastructure.files.storage import LocalStorage

async def add(data_path: Union[str, List[str]], dataset_name: str = None):
    if isinstance(data_path, str):
        # data_path is a data directory path
        if "data://" in data_path:
            return await add_data_directory(data_path.replace("data://", ""), dataset_name)
        # data_path is a file path
        if "file://" in data_path:
            return await add([data_path], dataset_name)
        # data_path is a text
        else:
            return await add_text(data_path, dataset_name)

    # data_path is a list of file paths
    return await add_files(data_path, dataset_name)

async def add_files(file_paths: List[str], dataset_name: str):
    data_directory_path = infrastructure_config.get_config()["data_path"]
    db_path = get_absolute_path("./data/cognee")
    db_location = f"{db_path}/cognee.duckdb"

    LocalStorage.ensure_directory_exists(db_path)

    processed_file_paths = []

    for file_path in file_paths:
        file_path = file_path.replace("file://", "")

        if data_directory_path not in file_path:
            file_name = file_path.split("/")[-1]
            dataset_file_path = data_directory_path + "/" + dataset_name.replace('.', "/") + "/" + file_name

            LocalStorage.copy_file(file_path, dataset_file_path)
            processed_file_paths.append(dataset_file_path)
        else:
            processed_file_paths.append(file_path)

    db = duckdb.connect(db_location)

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

                file_metadata = get_file_metadata(classified_data.get_data())

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

    return run_info

def extract_datasets_from_data(root_dir_path: str, parent_dir: str = "root"):
    datasets = {}

    root_dir_path = root_dir_path.replace("file://", "")

    for file_or_dir in listdir(root_dir_path):
        if path.isdir(path.join(root_dir_path, file_or_dir)):
            dataset_name = file_or_dir if parent_dir == "root" else parent_dir + "." + file_or_dir

            nested_datasets = extract_datasets_from_data("file://" + path.join(root_dir_path, file_or_dir), dataset_name)

            for dataset in nested_datasets.keys():
                datasets[dataset] = nested_datasets[dataset]
        else:
            if parent_dir not in datasets:
                datasets[parent_dir] = []

            datasets[parent_dir].append(path.join(root_dir_path, file_or_dir))

    return datasets

async def add_data_directory(data_path: str, dataset_name: str = None):
    datasets = extract_datasets_from_data(data_path)

    results = []

    for key in datasets.keys():
        if dataset_name is None or key.startswith(dataset_name):
            results.append(add(datasets[key], dataset_name = key))

    return await asyncio.gather(*results)

async def add_text(text: str, dataset_name: str):
    data_directory_path = infrastructure_config.get_config()["data_path"]

    classified_data = ingestion.classify(text)
    data_id = ingestion.identify(classified_data)

    storage_path = data_directory_path + "/" + dataset_name.replace(".", "/")
    LocalStorage.ensure_directory_exists(storage_path)

    text_file_name = str(data_id) + ".txt"
    LocalStorage(storage_path).store(text_file_name, classified_data.get_data())

    return await add(["file://" + storage_path + "/" + text_file_name], dataset_name)
