import os
import asyncio
import argparse
from cognee.tasks.repo_processor.get_repo_file_dependencies import get_repo_file_dependencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path", help="Path to the repository")
    args = parser.parse_args()

    repo_path = args.repo_path
    if not os.path.exists(repo_path):
        print(f"Error: The provided repository path does not exist: {repo_path}")
        return

    graph = asyncio.run(get_repo_file_dependencies(repo_path))

    for node in graph.nodes:
        print(f"Node: {node}")
        edges = graph.edges(node, data=True)
        for _, target, data in edges:
            print(f"  Edge to {target}, Relation: {data.get('relation')}")


if __name__ == "__main__":
    main()
