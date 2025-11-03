import os
import asyncio
import cognee
import pathlib
import subprocess

from cognee.shared.logging_utils import get_logger


logger = get_logger()


async def concurrent_subprocess_access():
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

    subprocess_directory_path = str(
        pathlib.Path(os.path.join(pathlib.Path(__file__).parent, "subprocesses/")).resolve()
    )

    writer_path = subprocess_directory_path + "/writer.py"
    reader_path = subprocess_directory_path + "/reader.py"

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    writer_process = subprocess.Popen([os.sys.executable, str(writer_path)])

    reader_process = subprocess.Popen([os.sys.executable, str(reader_path)])

    # Wait for both processes to complete
    writer_process.wait()
    reader_process.wait()

    logger.info("Basic write read subprocess example finished")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = """
            This is the text of the first cognify subprocess
            """
    await cognee.add(text, dataset_name="first_cognify_dataset")

    text = """
            This is the text of the second cognify subprocess
            """
    await cognee.add(text, dataset_name="second_cognify_dataset")

    first_cognify_path = subprocess_directory_path + "/simple_cognify_1.py"
    second_cognify_path = subprocess_directory_path + "/simple_cognify_2.py"

    first_cognify_process = subprocess.Popen([os.sys.executable, str(first_cognify_path)])

    second_cognify_process = subprocess.Popen([os.sys.executable, str(second_cognify_path)])

    # Wait for both processes to complete
    first_cognify_process.wait()
    second_cognify_process.wait()

    logger.info("Database concurrent subprocess example finished")


if __name__ == "__main__":
    asyncio.run(concurrent_subprocess_access())
