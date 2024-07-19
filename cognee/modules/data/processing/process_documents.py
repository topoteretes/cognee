from cognee.infrastructure.databases.graph import get_graph_engine
from .document_types import Document

async def process_documents(documents: list[Document], parent_node_id: str = None):
    graph_engine = await get_graph_engine()

    nodes = []
    edges = []

    if parent_node_id and await graph_engine.extract_node(parent_node_id) is None:
        nodes.append((parent_node_id, {}))

    document_nodes = await graph_engine.extract_nodes([str(document.id) for document in documents])

    for (document_index, document) in enumerate(documents):
        document_node = document_nodes[document_index] if document_index in document_nodes else None

        if document_node is None:
            nodes.append((str(document.id), document.to_dict()))

            if parent_node_id:
                edges.append((
                  parent_node_id,
                  str(document.id),
                  "has_document",
                  dict(relationship_name = "has_document"),
                ))

    if len(nodes) > 0:
        await graph_engine.add_nodes(nodes)
        await graph_engine.add_edges(edges)

    for document in documents:
        document_reader = document.get_reader()

        for document_chunk in document_reader.read(max_chunk_size = 1024):
            yield document_chunk
