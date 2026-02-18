import os
import asyncio
import pathlib
from uuid import UUID

import cognee
from cognee.shared.logging_utils import setup_logging, ERROR
from cognee.api.v1.datasets import datasets
from cognee.modules.users.methods import get_default_user


async def main():
    # Set data and system directory paths
    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent, ".data_storage/test_delete_data_and_dataset_if_empty"
            )
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".cognee_system/test_delete_data_and_dataset_if_empty",
            )
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    # Add the text, and make it available for cognify
    res_add_1 = await cognee.add(text, "nlp_dataset")
    res_add_2 = await cognee.add(
        "Quantum computing is the study of quantum computers.", "quantum_dataset"
    )

    # Use LLMs and cognee to create knowledge graph
    ret_val = await cognee.cognify()
    user = await get_default_user()

    for i, val in enumerate(ret_val):
        data_id = (
            res_add_1.data_ingestion_info[0]["data_id"]
            if i == 0
            else res_add_2.data_ingestion_info[0]["data_id"]
        )
        dataset_id = str(val)
        vector_db_path = os.path.join(
            cognee_directory_path, "databases", str(user.id), dataset_id + ".lance.db"
        )
        graph_db_path = os.path.join(
            cognee_directory_path, "databases", str(user.id), dataset_id + ".pkl"
        )

        # Check if databases are properly created and exist before deletion
        assert os.path.exists(graph_db_path), "Graph database file not found."
        assert os.path.exists(vector_db_path), "Vector database file not found."

        await datasets.delete_data(
            dataset_id=UUID(dataset_id),
            data_id=data_id,
            delete_dataset_if_empty=False,
        )

        # Confirm databases have NOT been deleted
        assert os.path.exists(graph_db_path), "Graph database file found."
        assert os.path.exists(vector_db_path), "Vector database file found."

        await datasets.delete_data(
            dataset_id=UUID(dataset_id),
            data_id=data_id,
            delete_dataset_if_empty=True,
        )

        # Confirm databases have been deleted
        assert not os.path.exists(graph_db_path), "Graph database file found."
        assert not os.path.exists(vector_db_path), "Vector database file found."


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
