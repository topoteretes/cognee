import asyncio
import math
import os

# from concurrent.futures import ProcessPoolExecutor
from typing import AsyncGenerator
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile, Repository


async def get_source_code_files(repo_path):
    """Get .py files and their source code"""
    if not os.path.exists(repo_path):
        return {}

    py_files_paths = (
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path)
        for file in files
        if (
            file.endswith(".py")
            and not file.startswith("test_")
            and not file.endswith("_test")
            and ".venv" not in file
        )
    )

    source_code_files = set()
    for file_path in py_files_paths:
        file_path = os.path.abspath(file_path)

        if os.path.getsize(file_path) == 0:
            continue

        source_code_files.add(file_path)

    return list(source_code_files)


def run_coroutine(coroutine_func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine_func(*args, **kwargs))
    loop.close()
    return result


async def get_repo_file_dependencies(
    repo_path: str, detailed_extraction: bool = False
) -> AsyncGenerator[DataPoint, None]:
    """Generate a dependency graph for Python files in the given repository path."""

    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path {repo_path} does not exist.")

    source_code_files = await get_source_code_files(repo_path)

    repo = Repository(
        id=uuid5(NAMESPACE_OID, repo_path),
        path=repo_path,
    )

    yield repo

    chunk_size = 100
    number_of_chunks = math.ceil(len(source_code_files) / chunk_size)
    chunk_ranges = [
        (
            chunk_number * chunk_size,
            min((chunk_number + 1) * chunk_size, len(source_code_files)) - 1,
        )
        for chunk_number in range(number_of_chunks)
    ]

    # Codegraph dependencies are not installed by default, so we import where we use them.
    from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies

    for start_range, end_range in chunk_ranges:
        # with ProcessPoolExecutor(max_workers=12) as executor:
        tasks = [
            get_local_script_dependencies(repo_path, file_path, detailed_extraction)
            for file_path in source_code_files[start_range : end_range + 1]
        ]

        results: list[CodeFile] = await asyncio.gather(*tasks)

        for source_code_file in results:
            source_code_file.part_of = repo

            yield source_code_file
