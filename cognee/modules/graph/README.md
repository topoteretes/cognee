# Layered Knowledge Graph

This module provides a simplified implementation of a layered knowledge graph, which allows organizing nodes and edges into hierarchical layers.

## Features

- **Hierarchical Layer Structure**: Organize your graph into layers with parent-child relationships
- **Cumulative Views**: Access nodes and edges from a layer and all its parent layers
- **Adapter-based Design**: Connect to different database backends using adapter pattern
- **NetworkX Integration**: Built-in support for NetworkX graph database
- **Type Safety**: Pydantic models ensure type safety and data validation
- **Async API**: All methods are async for better performance

## Components

- **GraphNode**: A node in the graph with a name, type, properties, and metadata
- **GraphEdge**: An edge connecting two nodes with an edge type, properties, and metadata
- **GraphLayer**: A layer in the graph that can contain nodes and edges, and can have parent layers
- **LayeredKnowledgeGraph**: The main graph class that manages layers, nodes, and edges

## Usage Example

```python
import asyncio
from uuid import UUID
from cognee.modules.graph.simplified_layered_graph import LayeredKnowledgeGraph
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter

async def main():
    # Initialize adapter
    adapter = NetworkXAdapter(filename="graph.pkl")
    await adapter.create_empty_graph("graph.pkl")
    
    # Create graph
    graph = LayeredKnowledgeGraph.create_empty("My Knowledge Graph")
    graph.set_adapter(LayeredGraphDBAdapter(adapter))
    
    # Add layers with parent-child relationships
    base_layer = await graph.add_layer(
        name="Base Layer", 
        description="Foundation concepts",
        layer_type="base"
    )
    
    derived_layer = await graph.add_layer(
        name="Derived Layer",
        description="Concepts built upon the base layer",
        layer_type="derived",
        parent_layers=[base_layer.id]  # Parent-child relationship
    )
    
    # Add nodes to layers
    node1 = await graph.add_node(
        name="Concept A",
        node_type="concept",
        properties={"importance": "high"},
        layer_id=base_layer.id
    )
    
    node2 = await graph.add_node(
        name="Concept B",
        node_type="concept",
        properties={"importance": "medium"},
        layer_id=derived_layer.id
    )
    
    # Connect nodes with an edge
    edge = await graph.add_edge(
        source_id=node1.id,
        target_id=node2.id,
        edge_type="RELATES_TO",
        properties={"strength": "high"},
        layer_id=derived_layer.id
    )
    
    # Get cumulative view (including parent layers)
    nodes, edges = await graph.get_cumulative_layer_graph(derived_layer.id)
    
    print(f"Nodes in cumulative view: {[n.name for n in nodes]}")
    print(f"Edges in cumulative view: {[e.edge_type for e in edges]}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Design Improvements

The simplified layered graph implementation offers several improvements over the previous approach:

1. **Clear Separation of Concerns**: In-memory operations vs. database operations
2. **More Intuitive API**: Methods have clear, consistent signatures
3. **Better Error Handling**: Comprehensive validation and error reporting
4. **Enhanced Debugging**: Detailed logging throughout
5. **Improved Caching**: Local caches reduce database load
6. **Method Naming Consistency**: All methods follow consistent naming conventions
7. **Reduced Complexity**: Simpler implementation with equivalent functionality

## Best Practices

- Always use the adapter pattern for database operations
- Use the provided factory methods for creating nodes and edges
- Leverage parent-child relationships for organizing related concepts
- Utilize cumulative views to access inherited nodes and edges
- Consider layer types for additional semantic meaning
- Use properties and metadata for storing additional information 