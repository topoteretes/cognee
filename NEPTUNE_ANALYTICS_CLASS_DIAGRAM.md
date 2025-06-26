# Neptune Analytics Class Diagram

## SUMMARY

This diagram shows the Neptune Analytics adapter class diagram. 

Benefits:
• Maintains consistency with existing Cognee patterns
• Follows the same interface contract as other graph database adapters
• Seamless integration with the factory pattern
• Proper separation of concerns with external library dependencies

The diagram shows how all components work together while maintaining the modular architecture that Cognee already uses for its graph database abstraction layer.

## ELEMENTS
• NeptuneAnalyticsAdapter extends GraphDBInterface just like the existing Neo4jAdapter
• URL format handling for neptune-graph://<GRAPH_ID>
• Integration with aws_langchain library through NeptuneAnalyticsClient
• Updated GraphEngineFactory to support the new provider
• Comprehensive test suite similar to the existing Neo4j tests

## PRESENTATION

```mermaid
classDiagram
    class GraphDBInterface {
        <<interface>>
        +query(query_input: str) dict
        +add_node(node: dict) dict
        +delete_node(node_id: str) bool
        +get_node(node_id: str) dict
        +add_edge(from_node: str, to_node: str, edge_data: dict) dict
        +has_edge(from_node: str, to_node: str) bool
        +has_edges(node_id: str) bool
        +add_nodes(nodes: list) list
        +delete_nodes(node_ids: list) bool
        +get_nodes(node_ids: list) list
        +add_edges(edges: list) list
        +delete_graph() bool
        +get_edges(edge_ids: list) list
        +get_graph_data() dict
        +get_graph_metrics() dict
        +get_neighbors(node_id: str) list
        +get_nodeset_subgraph(node_ids: list) dict
        +get_connections(from_nodes: list, to_nodes: list) list
    }

    class Neo4jAdapter {
        -driver: Neo4jDriver
        -database: str
        +query(query_input: str) dict
        +add_node(node: dict) dict
        +delete_node(node_id: str) bool
        +get_node(node_id: str) dict
        +add_edge(from_node: str, to_node: str, edge_data: dict) dict
        +has_edge(from_node: str, to_node: str) bool
        +has_edges(node_id: str) bool
        +add_nodes(nodes: list) list
        +delete_nodes(node_ids: list) bool
        +get_nodes(node_ids: list) list
        +add_edges(edges: list) list
        +delete_graph() bool
        +get_edges(edge_ids: list) list
        +get_graph_data() dict
        +get_graph_metrics() dict
        +get_neighbors(node_id: str) list
        +get_nodeset_subgraph(node_ids: list) dict
        +get_connections(from_nodes: list, to_nodes: list) list
    }

    class NeptuneAnalyticsAdapter {
        -graph_id: str
        -region: str
        -aws_credentials: dict
        -neptune_client: NeptuneAnalyticsClient
        +__init__(graph_database_url: str, **kwargs)
        +query(query_input: str) dict
        +add_node(node: dict) dict
        +delete_node(node_id: str) bool
        +get_node(node_id: str) dict
        +add_edge(from_node: str, to_node: str, edge_data: dict) dict
        +has_edge(from_node: str, to_node: str) bool
        +has_edges(node_id: str) bool
        +add_nodes(nodes: list) list
        +delete_nodes(node_ids: list) bool
        +get_nodes(node_ids: list) list
        +add_edges(edges: list) list
        +delete_graph() bool
        +get_edges(edge_ids: list) list
        +get_graph_data() dict
        +get_graph_metrics() dict
        +get_neighbors(node_id: str) list
        +get_nodeset_subgraph(node_ids: list) dict
        +get_connections(from_nodes: list, to_nodes: list) list
        -_parse_neptune_url(url: str) dict
        -_initialize_neptune_client() NeptuneAnalyticsClient
    }

    class NeptuneAnalyticsClient {
        <<aws_langchain>>
        +execute_query(query: str) dict
        +get_graph_summary() dict
        +create_node(properties: dict) dict
        +create_edge(from_id: str, to_id: str, properties: dict) dict
        +delete_node(node_id: str) bool
        +delete_edge(edge_id: str) bool
    }

    class GraphEngineFactory {
        +create_graph_engine(graph_database_provider: str, graph_database_url: str) GraphDBInterface
        -_create_neo4j_adapter(url: str) Neo4jAdapter
        -_create_neptune_analytics_adapter(url: str) NeptuneAnalyticsAdapter
        -_parse_database_url(url: str) dict
    }

    GraphDBInterface <|-- Neo4jAdapter
    GraphDBInterface <|-- NeptuneAnalyticsAdapter
    NeptuneAnalyticsAdapter --> NeptuneAnalyticsClient : uses
    GraphEngineFactory --> Neo4jAdapter : creates
    GraphEngineFactory --> NeptuneAnalyticsAdapter : creates
```

## Key Components

### NeptuneAnalyticsAdapter
- **Location**: `cognee/infrastructure/databases/graph/neptune_analytics_driver/adapter.py`
- **Purpose**: Main adapter class implementing GraphDBInterface for Neptune Analytics
- **Key Features**:
  - Parses `neptune-graph://<GRAPH_ID>` URL format
  - Uses aws_langchain library for Neptune Analytics operations
  - Implements all required CRUD and bulk operations
  - Handles AWS credentials and region configuration

### GraphEngineFactory Updates
- **Location**: `cognee/infrastructure/databases/graph/get_graph_engine.py`
- **Changes**: 
  - Add Neptune Analytics provider support
  - Handle neptune-graph:// URL parsing
  - Instantiate NeptuneAnalyticsAdapter when selected

### Test Suite
- **Location**: `cognee/tests/test_neptune_analytics.py`
- **Purpose**: Comprehensive testing similar to existing Neo4j tests
- **Features**:
  - Unit tests with mocks
  - Integration tests for real Neptune instances
  - URL parsing validation
  - Error handling verification

### Dependencies
- **aws_langchain**: Primary library for Neptune Analytics integration
- **boto3**: AWS SDK for authentication and region handling
- **pytest**: Testing framework consistency with existing tests
