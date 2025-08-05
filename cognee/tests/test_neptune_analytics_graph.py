import os
from dotenv import load_dotenv
import asyncio
from cognee.infrastructure.databases.graph.neptune_driver import NeptuneGraphDB
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.data.processing.document_types import TextDocument

# Set up Amazon credentials in .env file and get the values from environment variables
load_dotenv()
graph_id = os.getenv("GRAPH_ID", "")

na_adapter = NeptuneGraphDB(graph_id)


def setup():
    # Define nodes data before the main function
    # These nodes were defined using openAI from the following prompt:

    # Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads
    #     that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It
    #     complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load
    #     the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's
    #     stored in Amazon S3.

    document = TextDocument(
        name="text_test.txt",
        raw_data_location="git/cognee/examples/database_examples/data_storage/data/text_test.txt",
        external_metadata="{}",
        mime_type="text/plain",
    )
    document_chunk = DocumentChunk(
        text="Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads \n    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It \n    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load \n    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's \n    stored in Amazon S3.\n    ",
        chunk_size=187,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )

    graph_database = EntityType(name="graph database", description="graph database")
    neptune_analytics_entity = Entity(
        name="neptune analytics",
        description="A memory-optimized graph database engine for analytics that processes large amounts of graph data quickly.",
    )
    neptune_database_entity = Entity(
        name="amazon neptune database",
        description="A popular managed graph database that complements Neptune Analytics.",
    )

    storage = EntityType(name="storage", description="storage")
    storage_entity = Entity(
        name="amazon s3",
        description="A storage service provided by Amazon Web Services that allows storing graph data.",
    )

    nodes_data = [
        document,
        document_chunk,
        graph_database,
        neptune_analytics_entity,
        neptune_database_entity,
        storage,
        storage_entity,
    ]

    edges_data = [
        (
            str(document_chunk.id),
            str(storage_entity.id),
            "contains",
        ),
        (
            str(storage_entity.id),
            str(storage.id),
            "is_a",
        ),
        (
            str(document_chunk.id),
            str(neptune_database_entity.id),
            "contains",
        ),
        (
            str(neptune_database_entity.id),
            str(graph_database.id),
            "is_a",
        ),
        (
            str(document_chunk.id),
            str(document.id),
            "is_part_of",
        ),
        (
            str(document_chunk.id),
            str(neptune_analytics_entity.id),
            "contains",
        ),
        (
            str(neptune_analytics_entity.id),
            str(graph_database.id),
            "is_a",
        ),
    ]

    return nodes_data, edges_data


async def pipeline_method():
    """
    Example script using the neptune analytics with small sample data

    This example demonstrates how to add nodes to Neptune Analytics
    """

    print("------TRUNCATE GRAPH-------")
    await na_adapter.delete_graph()

    print("------SETUP DATA-------")
    nodes, edges = setup()

    print("------ADD NODES-------")
    await na_adapter.add_node(nodes[0])
    await na_adapter.add_nodes(nodes[1:])

    print("------GET NODES FROM DATA-------")
    node_ids = [str(node.id) for node in nodes]
    db_nodes = await na_adapter.get_nodes(node_ids)

    print("------RESULTS:-------")
    for n in db_nodes:
        print(n)

    print("------ADD EDGES-------")
    await na_adapter.add_edge(edges[0][0], edges[0][1], edges[0][2])
    await na_adapter.add_edges(edges[1:])

    print("------HAS EDGES-------")
    has_edge = await na_adapter.has_edge(
        edges[0][0],
        edges[0][1],
        edges[0][2],
    )
    if has_edge:
        print(f"found edge ({edges[0][0]})-[{edges[0][2]}]->({edges[0][1]})")

    has_edges = await na_adapter.has_edges(edges)
    if len(has_edges) > 0:
        print(f"found edges: {len(has_edges)} (expected: {len(edges)})")
    else:
        print(f"no edges found (expected: {len(edges)})")

    print("------GET GRAPH-------")
    all_nodes, all_edges = await na_adapter.get_graph_data()
    print(f"found {len(all_nodes)} nodes and found {len(all_edges)} edges")

    print("------NEIGHBORING NODES-------")
    center_node = nodes[2]
    neighbors = await na_adapter.get_neighbors(str(center_node.id))
    print(f'found {len(neighbors)} neighbors for node "{center_node.name}"')
    for neighbor in neighbors:
        print(neighbor)

    print("------NEIGHBORING EDGES-------")
    center_node = nodes[2]
    neighbouring_edges = await na_adapter.get_edges(str(center_node.id))
    print(f'found {len(neighbouring_edges)} edges neighbouring node "{center_node.name}"')
    for edge in neighbouring_edges:
        print(edge)

    print("------GET CONNECTIONS (SOURCE NODE)-------")
    document_chunk_node = nodes[0]
    connections = await na_adapter.get_connections(str(document_chunk_node.id))
    print(f'found {len(connections)} connections for node "{document_chunk_node.type}"')
    for connection in connections:
        src, relationship, tgt = connection
        src = src.get("name", src.get("type", "unknown"))
        relationship = relationship["relationship_name"]
        tgt = tgt.get("name", tgt.get("type", "unknown"))
        print(f'"{src}"-[{relationship}]->"{tgt}"')

    print("------GET CONNECTIONS (TARGET NODE)-------")
    connections = await na_adapter.get_connections(str(center_node.id))
    print(f'found {len(connections)} connections for node "{center_node.name}"')
    for connection in connections:
        src, relationship, tgt = connection
        src = src.get("name", src.get("type", "unknown"))
        relationship = relationship["relationship_name"]
        tgt = tgt.get("name", tgt.get("type", "unknown"))
        print(f'"{src}"-[{relationship}]->"{tgt}"')

    print("------SUBGRAPH-------")
    node_names = ["neptune analytics", "amazon neptune database"]
    subgraph_nodes, subgraph_edges = await na_adapter.get_nodeset_subgraph(Entity, node_names)
    print(
        f"found {len(subgraph_nodes)} nodes and  {len(subgraph_edges)} edges in the subgraph around {node_names}"
    )
    for subgraph_node in subgraph_nodes:
        print(subgraph_node)
    for subgraph_edge in subgraph_edges:
        print(subgraph_edge)

    print("------STAT-------")
    stat = await na_adapter.get_graph_metrics(include_optional=True)
    assert type(stat) is dict
    assert stat["num_nodes"] == 7
    assert stat["num_edges"] == 7
    assert stat["mean_degree"] == 2.0
    assert round(stat["edge_density"], 3) == 0.167
    assert stat["num_connected_components"] == [7]
    assert stat["sizes_of_connected_components"] == 1
    assert stat["num_selfloops"] == 0
    # Unsupported optional metrics
    assert stat["diameter"] == -1
    assert stat["avg_shortest_path_length"] == -1
    assert stat["avg_clustering"] == -1

    print("------DELETE-------")
    # delete all nodes and edges:
    await na_adapter.delete_graph()

    # delete all nodes by node id
    # node_ids = [str(node.id) for node in nodes]
    # await na_adapter.delete_nodes(node_ids)

    has_edges = await na_adapter.has_edges(edges)
    if len(has_edges) == 0:
        print("Delete successful")
    else:
        print("Delete failed")


async def misc_methods():
    print("------TRUNCATE GRAPH-------")
    await na_adapter.delete_graph()

    print("------SETUP TEST ENV-------")
    nodes, edges = setup()
    await na_adapter.add_nodes(nodes)
    await na_adapter.add_edges(edges)

    print("------GET GRAPH-------")
    all_nodes, all_edges = await na_adapter.get_graph_data()
    print(f"found {len(all_nodes)} nodes and found {len(all_edges)} edges")

    print("------GET DISCONNECTED-------")
    nodes_disconnected = await na_adapter.get_disconnected_nodes()
    print(nodes_disconnected)
    assert len(nodes_disconnected) == 0

    print("------Get Labels (Node)-------")
    node_labels = await na_adapter.get_node_labels_string()
    print(node_labels)

    print("------Get Labels (Edge)-------")
    edge_labels = await na_adapter.get_relationship_labels_string()
    print(edge_labels)

    print("------Get Filtered Graph-------")
    filtered_nodes, filtered_edges = await na_adapter.get_filtered_graph_data(
        [{"name": ["text_test.txt"]}]
    )
    print(filtered_nodes, filtered_edges)

    print("------Get Degree one nodes-------")
    degree_one_nodes = await na_adapter.get_degree_one_nodes("EntityType")
    print(degree_one_nodes)

    print("------Get Doc sub-graph-------")
    doc_sub_graph = await na_adapter.get_document_subgraph("test.txt")
    print(doc_sub_graph)

    print("------Fetch and Remove connections (Predecessors)-------")
    # Fetch test edge
    (src_id, dest_id, relationship) = edges[0]
    nodes_predecessors = await na_adapter.get_predecessors(node_id=dest_id, edge_label=relationship)
    assert len(nodes_predecessors) > 0

    await na_adapter.remove_connection_to_predecessors_of(
        node_ids=[src_id], edge_label=relationship
    )
    nodes_predecessors_after = await na_adapter.get_predecessors(
        node_id=dest_id, edge_label=relationship
    )
    # Return empty after relationship being deleted.
    assert len(nodes_predecessors_after) == 0

    print("------Fetch and Remove connections (Successors)-------")
    _, edges_suc = await na_adapter.get_graph_data()
    (src_id, dest_id, relationship, _) = edges_suc[0]

    nodes_successors = await na_adapter.get_successors(node_id=src_id, edge_label=relationship)
    assert len(nodes_successors) > 0

    await na_adapter.remove_connection_to_successors_of(node_ids=[dest_id], edge_label=relationship)
    nodes_successors_after = await na_adapter.get_successors(
        node_id=src_id, edge_label=relationship
    )
    assert len(nodes_successors_after) == 0

    # no-op
    await na_adapter.project_entire_graph()
    await na_adapter.drop_graph()
    await na_adapter.graph_exists()

    pass


if __name__ == "__main__":
    asyncio.run(pipeline_method())
    asyncio.run(misc_methods())
