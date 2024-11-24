import os
import networkx as nx

from cognee.shared.CodeGraphEntities import CodeFile, CodeRelationship, Repository
from cognee.tasks.storage import add_data_points


async def convert_graph_from_code_graph(
    graph: nx.DiGraph, repo_path: str
) -> tuple[str, list[CodeFile], list[CodeRelationship]]:
    code_objects = code_objects_from_di_graph(graph, repo_path)

    add_data_points(code_objects)

    return code_objects


def create_code_file(path, type, repo):
    abspath = os.path.abspath(path)

    with open(abspath, "r") as f:
        source_code = f.read()

    code_file = CodeFile(
        extracted_id = abspath,
        type = type,
        source_code = source_code,
        part_of = repo,
    )

    return code_file


def code_objects_from_di_graph(
    graph: nx.DiGraph, repo_path: str
) -> tuple[Repository, list[CodeFile], list[CodeRelationship]]:
    repo = Repository(path=repo_path)

    code_files = [
        create_code_file(os.path.join(repo_path, path), "python_file", repo)
        for path in graph.nodes
    ]

    code_relationships = [
        CodeRelationship(
            os.path.join(repo_path, source),
            os.path.join(repo_path, target),
            "python_file",
            graph.get_edge_data(source, target)["relation"],
        )
        for source, target in graph.edges
    ]

    return (repo, code_files, code_relationships)
