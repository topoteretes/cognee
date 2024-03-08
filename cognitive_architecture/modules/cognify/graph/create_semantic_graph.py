from cognitive_architecture.infrastructure.graph.get_graph_client import get_graph_client




def create_semantic_graph(
    text_input: str,
    filename: str,
    context,
    response_model: Type[BaseModel]
) -> KnowledgeGraph:
    graph_type = GraphDBType.NEO4J

    # Call the get_graph_client function with the selected graph type
    graph_client = get_graph_client(graph_type)

GraphDBInterface