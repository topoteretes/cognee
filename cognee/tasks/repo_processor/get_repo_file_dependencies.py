import os
from typing import AsyncGenerator
from uuid import NAMESPACE_OID, uuid5
import aiofiles
from tqdm.asyncio import tqdm

from cognee.infrastructure.engine import DataPoint
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
        for root, _, files in os.walk(repo_path) for file in files if file.endswith(".py")
    )

    py_files_dict = {}
    for file_path in py_files_paths:
        absolute_path = os.path.abspath(file_path)
        relative_path, source_code = await get_py_path_and_source(absolute_path)
        py_files_dict[relative_path] = {"source_code": source_code}

    return py_files_dict


def get_edge(file_path: str, dependency: str, repo_path: str, relative_paths: bool = False) -> tuple:
    if relative_paths:
        file_path = os.path.relpath(file_path, repo_path)
        dependency = os.path.relpath(dependency, repo_path)
    return (file_path, dependency, {"relation": "depends_directly_on"})


async def get_repo_file_dependencies(repo_path: str) -> AsyncGenerator[list[DataPoint], None]:
    """Generate a dependency graph for Python files in the given repository path."""
    py_files_dict = await get_py_files_dict(repo_path)

    repo = Repository(
        id = uuid5(NAMESPACE_OID, repo_path),
        path = repo_path,
    )

    # data_points = [repo]
    yield repo

    # dependency_graph = nx.DiGraph()

    # dependency_graph.add_nodes_from(py_files_dict.items())

    async for file_path, metadata in tqdm(py_files_dict.items(), desc="Repo dependency graph", unit="file"):
        source_code = metadata.get("source_code")
        if source_code is None:
            continue

        dependencies = await get_local_script_dependencies(os.path.join(repo_path, file_path), repo_path)

        # data_points.append()
        yield CodeFile(
            id = uuid5(NAMESPACE_OID, file_path),
            source_code = source_code,
            extracted_id = file_path,
            part_of = repo,
            depends_on = [
                CodeFile(
                    id = uuid5(NAMESPACE_OID, dependency),
                    extracted_id = dependency,
                    part_of = repo,
                ) for dependency in dependencies
            ] if len(dependencies) else None,
        )
        # dependency_edges = [get_edge(file_path, dependency, repo_path) for dependency in dependencies]

        # dependency_graph.add_edges_from(dependency_edges)

    # return data_points
    # return dependency_graph
