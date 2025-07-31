import asyncio
import math
import os
import fnmatch
from typing import AsyncGenerator, Optional, List
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile, Repository


async def get_source_code_files(repo_path: str, excluded_paths: Optional[List[str]] = None):
    """
    Retrieve Python source code files from the specified repository path,
    excluding paths and file patterns commonly irrelevant to code analysis.

    Parameters:
    -----------
    - repo_path: Root path of the repository to search
    - excluded_paths: Optional list of path fragments or glob patterns to exclude

    Returns:
    --------
    List of absolute file paths for .py files, excluding test files,
    empty files, and files under ignored directories or matching ignore patterns.
    """

    if not os.path.exists(repo_path):
        return []

    # Default exclusions
    default_excluded_patterns = [
        ".venv/", "venv/", "__pycache__/", ".pytest_cache/", "build/", "dist/",
        "node_modules/", ".npm/", ".git/", ".svn/", ".idea/", ".vscode/", "tmp/", "temp/",
        "*.pyc", "*.pyo", "*.log", "*.tmp"
    ]

    excluded_patterns = default_excluded_patterns + (excluded_paths or [])

    py_files_paths = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_path)

            # Check for exclusion
            should_exclude = any(
                pattern in rel_path or fnmatch.fnmatch(rel_path, pattern)
                for pattern in excluded_patterns
            )
            if should_exclude:
                continue

            if (
                file.endswith(".py")
                and not file.startswith("test_")
                and not file.endswith("_test")
            ):
                py_files_paths.append(full_path)

    source_code_files = set()
    for file_path in py_files_paths:
        file_path = os.path.abspath(file_path)
        if os.path.getsize(file_path) == 0:
            continue
        source_code_files.add(file_path)

    return list(source_code_files)


def run_coroutine(coroutine_func, *args, **kwargs):
    """
    Run a coroutine function until it completes.

    This function creates a new asyncio event loop, sets it as the current loop, and
    executes the given coroutine function with the provided arguments. Once the coroutine
    completes, the loop is closed.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine_func(*args, **kwargs))
    loop.close()
    return result


async def get_repo_file_dependencies(
    repo_path: str,
    detailed_extraction: bool = False,
    excluded_paths: Optional[List[str]] = None
) -> AsyncGenerator[DataPoint, None]:
    """
    Generate a dependency graph for Python files in the given repository path.

    Parameters:
    -----------
    - repo_path: Path to local repository
    - detailed_extraction: Whether to extract fine-grained dependencies
    - excluded_paths: Optional custom exclusion list
    """

    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path {repo_path} does not exist.")

    source_code_files = await get_source_code_files(repo_path, excluded_paths=excluded_paths)

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

    from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies

    for start_range, end_range in chunk_ranges:
        tasks = [
            get_local_script_dependencies(repo_path, file_path, detailed_extraction)
            for file_path in source_code_files[start_range : end_range + 1]
        ]

        results: list[CodeFile] = await asyncio.gather(*tasks)

        for source_code_file in results:
            source_code_file.part_of = repo
            yield source_code_file
