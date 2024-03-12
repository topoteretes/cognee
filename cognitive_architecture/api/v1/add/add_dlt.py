from typing import List, Union
from os import path, listdir
import asyncio
import dlt
import duckdb
from unstructured.cleaners.core import clean
from cognitive_architecture.root_dir import get_absolute_path
import cognitive_architecture.modules.ingestion as ingestion
from cognitive_architecture.infrastructure.files import get_file_metadata
from cognitive_architecture.infrastructure.files.storage import LocalStorage

async def add_dlt(file_paths: Union[str, List[str]], dataset_name: str = None):
    if isinstance(file_paths, str):
        # Directory path provided, we need to extract the file paths and dataset name

        def list_dir_files(root_dir_path: str, parent_dir: str = "root"):
            datasets = {}

            for file_or_dir in listdir(root_dir_path):
                if path.isdir(path.join(root_dir_path, file_or_dir)):
                    dataset_name = file_or_dir if parent_dir == "root" else parent_dir + "." + file_or_dir
                    dataset_name = clean(dataset_name.replace(" ", "_"))

                    nested_datasets = list_dir_files(path.join(root_dir_path, file_or_dir), dataset_name)

                    for dataset in nested_datasets:
                        datasets[dataset] = nested_datasets[dataset]
                else:
                    if parent_dir not in datasets:
                        datasets[parent_dir] = []

                    datasets[parent_dir].append(path.join(root_dir_path, file_or_dir))

            return datasets

        datasets = list_dir_files(file_paths)

        results = []

        for key in datasets:
            if dataset_name is not None and not key.startswith(dataset_name):
                continue

            results.append(add_dlt(datasets[key], dataset_name = key))

        return await asyncio.gather(*results)


    db_path = get_absolute_path("./data/cognee")
    db_location = f"{db_path}/cognee.duckdb"

    LocalStorage.ensure_directory_exists(db_path)

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
        data_resources(file_paths),
        table_name = "file_metadata",
        dataset_name = dataset_name,
        write_disposition = "merge",
    )

    return run_info
