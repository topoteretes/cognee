import os
import asyncio
import cognee
import pathlib
import subprocess

from cognee.infrastructure.databases.graph import get_graph_engine
from collections import Counter
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from multiprocessing import Process


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

    writer_process = subprocess.Popen([os.sys.executable, "writer.py"])

    reader_process = subprocess.Popen([os.sys.executable, "reader.py"])

    # Wait for both processes to complete
    writer_process.wait()
    reader_process.wait()


if __name__ == "__main__":
    asyncio.run(test_concurrent_subprocess_access())
