import asyncio
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.data.processing.document_types import TextDocument

async def main():
    print("Initializing...")
    
    # Create a NodeSet
    ns = NodeSet(name="first")
    print(f"NodeSet created: {ns.id}, {ns.name}")
    
    # Create Document to satisfy is_part_of
    doc = TextDocument(
        name="doc1",
        text="some text",
        raw_data_location="location",
        mime_type="text/plain",
        external_metadata="{}"
    )

    # Create DocumentChunk
    chunk = DocumentChunk(
        text="Sample text",
        chunk_size=10,
        chunk_index=0,
        cut_type="word",
        is_part_of=doc,
    )
    
    # Assign belongs_to_set
    chunk.belongs_to_set = [ns]
    
    print(f"Chunk created. belongs_to_set: {chunk.belongs_to_set}")
    
    # Run get_graph_from_model
    added_nodes = {}
    added_edges = {}
    
    print("Running get_graph_from_model...")
    nodes, edges = await get_graph_from_model(
        chunk,
        added_nodes=added_nodes,
        added_edges=added_edges
    )
    
    print(f"Result Nodes: {len(nodes)}")
    for n in nodes:
        print(f" - Node: Type={type(n).__name__}, ID={n.id}")
        if isinstance(n, NodeSet):
             print(f"   -> Found NodeSet: {n.name}")

    print(f"Result Edges: {len(edges)}")
    for e in edges:
        print(f" - Edge: {e}")

if __name__ == "__main__":
    asyncio.run(main())
