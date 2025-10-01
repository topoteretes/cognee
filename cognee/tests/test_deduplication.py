import hashlib
import os
from cognee.shared.logging_utils import get_logger
import pathlib
import pytest

import cognee
from cognee.infrastructure.databases.relational import get_relational_engine

logger = get_logger()


async def test_deduplication():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    relational_engine = get_relational_engine()

    dataset_name = "test_deduplication"
    dataset_name2 = "test_deduplication2"

    # Test deduplication of local files
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing_copy.txt"
    )
    await cognee.add([explanation_file_path], dataset_name, incremental_loading=False)
    await cognee.add([explanation_file_path2], dataset_name2, incremental_loading=False)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."
    assert result[0]["name"] == "Natural_language_processing_copy", (
        "Result name does not match expected value."
    )

    result = await relational_engine.get_all_data_from_table("datasets")
    assert len(result) == 2, "Unexpected number of datasets found."
    assert result[0]["name"] == dataset_name, "Result name does not match expected value."
    assert result[1]["name"] == dataset_name2, "Result name does not match expected value."

    result = await relational_engine.get_all_data_from_table("dataset_data")
    assert len(result) == 2, "Unexpected number of dataset data relationships found."
    assert result[0]["data_id"] == result[1]["data_id"], "Data item is not reused between datasets."
    assert result[0]["dataset_id"] != result[1]["dataset_id"], "Dataset items are not different."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of text input
    text = os.path.join(pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt")

    await cognee.add([text], dataset_name)
    await cognee.add([text], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."
    assert hashlib.md5(text.encode("utf-8")).hexdigest() in result[0]["name"], (
        "Content hash is not a part of file name."
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of image files
    explanation_file_path = os.path.join(pathlib.Path(__file__).parent, "test_data/example.png")
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/example_copy.png"
    )

    await cognee.add([explanation_file_path], dataset_name)
    await cognee.add([explanation_file_path2], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Test deduplication of sound files
    explanation_file_path = os.path.join(
        pathlib.Path(__file__).parent, "test_data/text_to_speech.mp3"
    )
    explanation_file_path2 = os.path.join(
        pathlib.Path(__file__).parent, "test_data/text_to_speech_copy.mp3"
    )

    await cognee.add([explanation_file_path], dataset_name)
    await cognee.add([explanation_file_path2], dataset_name2)

    result = await relational_engine.get_all_data_from_table("data")
    assert len(result) == 1, "More than one data entity was found."

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_deduplication_postgres():
    cognee.config.set_vector_db_config(
        {"vector_db_url": "", "vector_db_key": "", "vector_db_provider": "pgvector"}
    )
    cognee.config.set_relational_db_config(
        {
            "db_name": "cognee_db",
            "db_host": "127.0.0.1",
            "db_port": "5432",
            "db_username": "cognee",
            "db_password": "cognee",
            "db_provider": "postgres",
        }
    )

    await test_deduplication()


@pytest.mark.asyncio
async def test_deduplication_sqlite():
    cognee.config.set_vector_db_config(
        {"vector_db_url": "", "vector_db_key": "", "vector_db_provider": "lancedb"}
    )
    cognee.config.set_relational_db_config(
        {
            "db_provider": "sqlite",
        }
    )

    await test_deduplication()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_deduplication")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_deduplication")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await test_deduplication_postgres()
    await test_deduplication_sqlite()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
