import os
from dotenv import load_dotenv
import asyncio
from cognee.infrastructure.databases.graph.neptune_analytics_driver import NeptuneAnalyticsAdapter
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.data.processing.document_types import TextDocument
from cognee.infrastructure.engine import DataPoint
from infrastructure.databases.graph.graph_db_interface import EdgeData

# Set up Amazon credentials in .env file and get the values from environment variables
load_dotenv()
graph_id = os.getenv('GRAPH_ID', "")

na_adapter = NeptuneAnalyticsAdapter(graph_id)

def setup():
    # Define nodes data before the main function
    # These nodes were defined using openAI from the following prompt:

    # Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads
    #     that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It
    #     complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load
    #     the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's
    #     stored in Amazon S3.

    document = TextDocument(
        name='text.txt',
        raw_data_location='git/cognee/examples/database_examples/data_storage/data/text.txt',
        external_metadata='{}',
        mime_type='text/plain'
    )
    document_chunk = DocumentChunk(
        text="Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads \n    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It \n    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load \n    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's \n    stored in Amazon S3.\n    ",
        chunk_size=187,
        chunk_index=0,
        cut_type='paragraph_end',
        is_part_of=document,
    )

    graph_database = EntityType(name='graph database', description='graph database')
    neptune_analytics_entity = Entity(
        name='neptune analytics',
        description='A memory-optimized graph database engine for analytics that processes large amounts of graph data quickly.',
        is_type=graph_database
    )
    neptune_database_entity = Entity(
        name='amazon neptune database',
        description='A popular managed graph database that complements Neptune Analytics.',
        is_type=graph_database
    )

    storage = EntityType(name='storage', description='storage')
    storage_entity = Entity(
        name='amazon s3',
        description='A storage service provided by Amazon Web Services that allows storing graph data.',
        is_type=storage
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
            'contains',
        ),
        (
            str(storage_entity.id),
            str(storage.id),
            'is_a',
        ),
        (
            str(document_chunk.id),
            str(neptune_database_entity.id),
            'contains',
        ),
        (
            str(neptune_database_entity.id),
            str(graph_database.id),
            'is_a',
        ),
        (
            str(document_chunk.id),
            str(document.id),
            'is_part_of',
        ),
        (
            str(document_chunk.id),
            str(neptune_analytics_entity.id),
            'contains',
        ),
        (
            str(neptune_analytics_entity.id),
            str(graph_database.id),
            'is_a',
        ),
    ]

    return nodes_data, edges_data

async def main():
    """
    Example script using the neptune analytics with small sample data

    This example demonstrates how to add nodes to Neptune Analytics
    """

    print("------SETUP DATA-------")
    nodes, edges = setup()

    print("------ADD NODES-------")
    await na_adapter.add_nodes(nodes)

    print("------GET NODES FROM DATA-------")
    node_ids = [str(node.id) for node in nodes]
    db_nodes = await na_adapter.get_nodes(node_ids)

    print("------RESULTS:-------")
    for n in db_nodes:
        print(n)

    print("------ADD EDGES-------")
    await na_adapter.add_edges(edges)

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

    print("------NEIGHBORS-------")
    neighbors =  await na_adapter.get_neighbors(str(nodes[2].id))
    print(f"found {len(neighbors)} neighbors for node \"{nodes[2].name}\"")
    for neighbor in neighbors:
        print(neighbor)

    print("------SUBGRAPH-------")
    node_names = ["neptune analytics", "amazon neptune database"]
    subgraph_nodes, subgraph_edges = await na_adapter.get_nodeset_subgraph(Entity, node_names)
    print(f"found {len(subgraph_nodes)} nodes and  {len(subgraph_edges)} edges in the subgraph around {node_names}")
    for subgraph_node in subgraph_nodes:
        print(subgraph_node)
    for subgraph_edge in subgraph_edges:
        print(subgraph_edge)

    print("------DELETE-------")
    # delete all nodes and edges:
    # await na_adapter.delete_graph()

    # delete all nodes by node id
    node_ids = [str(node.id) for node in nodes]
    await na_adapter.delete_nodes(node_ids)

    has_edges = await na_adapter.has_edges(edges)
    if len(has_edges) == 0:
        print(f"Delete successful")
    else:
        print(f"Delete failed")

if __name__ == "__main__":
    asyncio.run(main())
