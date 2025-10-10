import os
import asyncio
import cognee
import pathlib

from cognee.infrastructure.databases.graph import get_graph_engine
from collections import Counter
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def test_concurrent_subprocess_access():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/concurrent_tasks")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/concurrent_tasks")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text1 = "Dave watches Dexter Resurrection"
    text2 = "Ana likes apples"
    text3 = "Bob prefers Cognee over other solutions"

    await cognee.add([text1, text2, text3], dataset_name="edge_ingestion_test")

    user = await get_default_user()

    await cognee.cognify(["edge_ingestion_test"], user=user)


if __name__ == "__main__":
    asyncio.run(test_concurrent_subprocess_access())
