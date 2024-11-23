import os
import aiofiles

import networkx as nx
from typing import Dict, List

from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies


async def get_py_path_and_source(file_path, repo_path):
    relative_path = os.path.relpath(file_path, repo_path)
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            source_code = await f.read()
        return relative_path, source_code
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return relative_path, None


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
        relative_path, source_code = await get_py_path_and_source(file_path, repo_path)
        py_files_dict[relative_path] = {"source_code": source_code}

    return py_files_dict

def get_edge(file_path: str, dependency: str, repo_path: str, relative_paths: bool = True) -> tuple:
    if relative_paths:
        file_path = os.path.relpath(file_path, repo_path)
        dependency = os.path.relpath(dependency, repo_path)
    return (file_path, dependency, {"relation": "depends_directly_on"})


async def get_repo_dependency_graph(repo_path: str) -> nx.DiGraph:
    """Generate a dependency graph for Python files in the given repository path."""
    py_files_dict = await get_py_files_dict(repo_path)

    dependency_graph = nx.DiGraph()

    dependency_graph.add_nodes_from(py_files_dict.items())

    for file_path, metadata in py_files_dict.items():
        source_code = metadata.get("source_code")
        if source_code is None:
            continue

        dependencies = await get_local_script_dependencies(file_path, repo_path)
        dependency_edges = [get_edge(file_path, dependency, repo_path) for dependency in dependencies]
        dependency_graph.add_edges_from(dependency_edges)
    return dependency_graph
