import os
import pathlib
import cognee
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_parallel_databases")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_parallel_databases")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(["TEST1"], "test1")
    await cognee.add(["TEST2"], "test2")

    task_1_config = {
        "vector_db_url": os.path.join(cognee_directory_path, "cognee1.test"),
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
    }
    task_2_config = {
        "vector_db_url": os.path.join(cognee_directory_path, "cognee2.test"),
        "vector_db_key": "",
        "vector_db_provider": "lancedb",
    }

    task_1_graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": os.path.join(cognee_directory_path, "kuzu1.db"),
    }
    task_2_graph_config = {
        "graph_database_provider": "kuzu",
        "graph_file_path": os.path.join(cognee_directory_path, "kuzu2.db"),
    }

    # schedule both cognify calls concurrently
    task1 = asyncio.create_task(
        cognee.cognify(
            ["test1"], vector_db_config=task_1_config, graph_db_config=task_1_graph_config
        )
    )
    task2 = asyncio.create_task(
        cognee.cognify(
            ["test2"], vector_db_config=task_2_config, graph_db_config=task_2_graph_config
        )
    )

    # wait until both are done (raises first error if any)
    await asyncio.gather(task1, task2)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
