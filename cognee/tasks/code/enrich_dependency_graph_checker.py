import os
import asyncio
import argparse
from cognee.tasks.repo_processor.get_repo_file_dependencies import get_repo_file_dependencies
from cognee.tasks.repo_processor.enrich_dependency_graph import enrich_dependency_graph


def main():
    """
    Execute the main logic of the dependency graph processor.

    This function sets up argument parsing to retrieve the repository path, checks the
    existence of the specified path, and processes the repository to produce a dependency
    graph. If the repository path does not exist, it logs an error message and terminates
    without further execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to the repository")
    args = parser.parse_args()

    repo_path = args.repo_path
    if not os.path.exists(repo_path):
        print(f"Error: The provided repository path does not exist: {repo_path}")
        return

    graph = asyncio.run(get_repo_file_dependencies(repo_path))
    graph = asyncio.run(enrich_dependency_graph(graph))
    for node in graph.nodes:
        print(f"Node: {node}")
        for _, target, data in graph.out_edges(node, data=True):
            print(f"  Edge to {target}, data: {data}")


if __name__ == "__main__":
    main()
