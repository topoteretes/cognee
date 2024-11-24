from uuid import UUID, uuid4
import os
import networkx as nx

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.CodeGraphEntities import CodeFile, CodeRelationship, Repository
from cognee.shared.data_models import Edge, KnowledgeGraph, Node
from cognee.tasks.storage import add_data_points


async def convert_graph_from_code_graph(
    graph: nx.DiGraph, repo_path: str
) -> tuple[str, list[CodeFile], list[CodeRelationship]]:

    repo, nodes, edges = code_objects_from_di_graph(graph, repo_path)

    graph_engine = await get_graph_engine()

    code_knowledge_graph = build_code_knowledge_graph(nodes, edges)
    repo_and_knowledge_graph = [(repo, code_knowledge_graph)]

    existing_edges_map = await retrieve_existing_edges(
        repo_and_knowledge_graph, graph_engine
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        repo_and_knowledge_graph, existing_edges_map
    )

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return nodes


def convert_node(node: CodeFile) -> Node:
    return Node(
        id=str(node.id),
        name=node.extracted_id,
        type=node.type,
        description=f"{node.source_code = }",
        properties={},
    )


def convert_edge(edge: CodeRelationship, extracted_ids_to_ids: dict[str, UUID]) -> Edge:
    return Edge(
        source_node_id=str(extracted_ids_to_ids[edge.source_id]),
        target_node_id=str(extracted_ids_to_ids[edge.target_id]),
        relationship_name=f"{edge.type}_{edge.relation}",
    )


def build_code_knowledge_graph(nodes: list[CodeFile], edges: list[CodeRelationship]):
    extracted_ids_to_ids = {node.extracted_id: node.id for node in nodes}
    graph_nodes = [convert_node(node) for node in nodes]
    graph_edges = [convert_edge(edge, extracted_ids_to_ids) for edge in edges]
    return KnowledgeGraph(nodes=graph_nodes, edges=graph_edges)


def create_code_file(path, type):
    abspath = os.path.abspath(path)
    print(f"{path = } - {abspath = }")
    with open(abspath, "r") as f:
        source_code = f.read()
    code_file = CodeFile(extracted_id=abspath, type=type, source_code=source_code)
    return (code_file, abspath)


def create_code_relationship(
    source_path: str, target_path: str, type: str, relation: str
):
    return CodeRelationship(
        source_id=source_path, target_id=target_path, type=type, relation=relation
    )


def code_objects_from_di_graph(
    graph: nx.DiGraph, repo_path: str
) -> tuple[Repository, list[CodeFile], list[CodeRelationship]]:
    repo = Repository(path=repo_path)

    code_files = [
        create_code_file(os.path.join(repo_path, path), "python_file")[0]
        for path in graph.nodes
    ]

    code_relationships = [
        create_code_relationship(
            os.path.join(repo_path, source),
            os.path.join(repo_path, target),
            "python_file",
            graph.get_edge_data(source, target, v)["relation"],
        )
        for source, target, v in graph.edges
    ]

    return (repo, code_files, code_relationships)
