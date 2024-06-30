from cognee.infrastructure.databases.graph import get_graph_engine
from .document_types import Document

async def process_documents(documents: list[Document], parent_node_id: str):
    graph_engine = await get_graph_engine()

    nodes = [(str(document.id), document.to_dict()) for document in documents]

    if await graph_engine.extract_node(parent_node_id) is None:
        nodes.append((parent_node_id, {}))

    await graph_engine.add_nodes(nodes)

    await graph_engine.add_edges([(
        parent_node_id,
        str(document.id),
        "has_document",
        dict(relationship_name = "has_document"),
    ) for document in documents])

    for document in documents:
        document_reader = document.get_reader()
        for document_chunk in document_reader.read(max_chunk_size = 1024):
            yield document_chunk
