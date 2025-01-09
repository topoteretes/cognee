import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import AsyncGenerator
from uuid import NAMESPACE_OID, uuid5

import aiofiles

from cognee.shared.CodeGraphEntities import CodeFile, Repository
from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies


async def get_py_path_and_source(file_path):
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            source_code = await f.read()
        return file_path, source_code
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return file_path, None


async def get_py_files_dict(repo_path):
    """Get .py files and their source code"""
    if not os.path.exists(repo_path):
        return {}

    py_files_paths = (
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path)
        for file in files
        if file.endswith(".py")
    )

    py_files_dict = {}
    for file_path in py_files_paths:
        absolute_path = os.path.abspath(file_path)

        if os.path.getsize(absolute_path) == 0:
            continue

        relative_path, source_code = await get_py_path_and_source(absolute_path)
        py_files_dict[relative_path] = {"source_code": source_code}

    return py_files_dict


def get_edge(
    file_path: str, dependency: str, repo_path: str, relative_paths: bool = False
) -> tuple:
    if relative_paths:
        file_path = os.path.relpath(file_path, repo_path)
        dependency = os.path.relpath(dependency, repo_path)
    return (file_path, dependency, {"relation": "depends_directly_on"})


def run_coroutine(coroutine_func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine_func(*args, **kwargs))
    loop.close()
    return result


async def get_repo_file_dependencies(repo_path: str) -> AsyncGenerator[list, None]:
    """Generate a dependency graph for Python files in the given repository path."""

    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path {repo_path} does not exist.")

    py_files_dict = await get_py_files_dict(repo_path)

    repo = Repository(
        id=uuid5(NAMESPACE_OID, repo_path),
        path=repo_path,
    )

    yield [repo]

    with ProcessPoolExecutor(max_workers=12) as executor:
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                executor,
                run_coroutine,
                get_local_script_dependencies,
                os.path.join(repo_path, file_path),
                repo_path,
            )
            for file_path, metadata in py_files_dict.items()
            if metadata.get("source_code") is not None
        ]

        results = await asyncio.gather(*tasks)

        code_files = []
        for (file_path, metadata), dependencies in zip(py_files_dict.items(), results):
            source_code = metadata.get("source_code")

            code_files.append(
                CodeFile(
                    id=uuid5(NAMESPACE_OID, file_path),
                    source_code=source_code,
                    extracted_id=file_path,
                    part_of=repo,
                    depends_on=[
                        CodeFile(
                            id=uuid5(NAMESPACE_OID, dependency),
                            extracted_id=dependency,
                            part_of=repo,
                            source_code=py_files_dict.get(dependency, {}).get("source_code"),
                        )
                        for dependency in dependencies
                    ]
                    if dependencies
                    else None,
                )
            )

        yield code_files
