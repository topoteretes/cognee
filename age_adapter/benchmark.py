import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from age_adapter.adapter import ApacheAGEAdapter
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from uuid import UUID


class SimpleNode(DataPoint):
    model_config = {"extra": "allow"}
    
    def __init__(self, node_id: str, properties: Dict[str, Any]):
        try:
            node_uuid = UUID(node_id) if '-' in node_id else UUID(int=hash(node_id) & ((1 << 128) - 1))
        except:
            node_uuid = UUID(int=hash(node_id) & ((1 << 128) - 1))
        super().__init__(id=node_uuid, **properties)


async def main():
    age_adapter = ApacheAGEAdapter(
        host="localhost",
        port=5432,
        username="cognee",
        password="cognee",
        database="cognee_db",
        graph_name="benchmark_graph"
    )
    
    neo4j_adapter = Neo4jAdapter(
        graph_database_url="bolt://localhost:7687",
        graph_database_username="neo4j",
        graph_database_password="pleaseletmein",
        graph_database_name=None
    )
    
    await age_adapter.connect()
    await neo4j_adapter.initialize()
    
    batch_size = 500
    node_ids = [f"node_{i}" for i in range(batch_size)]
    nodes = [(nid, ["TestNode"], {"name": f"Node {i}", "value": i}) 
            for i, nid in enumerate(node_ids)]
    
    await age_adapter.drop_graph(recreate=True)
    await neo4j_adapter.delete_graph()
    
    start = time.perf_counter()
    for node_id, labels, props in nodes:
        await age_adapter.add_node(node_id, labels, props)
    age_time_single_new = time.perf_counter() - start
    
    start = time.perf_counter()
    for node_id, labels, props in nodes:
        neo4j_node = SimpleNode(node_id, props)
        await neo4j_adapter.add_node(neo4j_node)
    neo4j_time_single_new = time.perf_counter() - start
    
    print(f"Node Ingestion Single (New): AGE={age_time_single_new:.4f}s, Neo4j={neo4j_time_single_new:.4f}s")
    
    half = batch_size // 2
    existing_nodes = nodes[:half]
    new_node_ids = [f"node_{i}" for i in range(batch_size, batch_size + half)]
    new_nodes = [(nid, ["TestNode"], {"name": f"Node {i}", "value": i}) 
                 for i, nid in enumerate(new_node_ids, start=batch_size)]
    merge_nodes = existing_nodes + new_nodes
    
    for node_id, labels, props in existing_nodes:
        await age_adapter.add_node(node_id, labels, props)
        neo4j_node = SimpleNode(node_id, props)
        await neo4j_adapter.add_node(neo4j_node)
    
    start = time.perf_counter()
    for node_id, labels, props in merge_nodes:
        await age_adapter.add_node(node_id, labels, props)
    age_time_single_merge = time.perf_counter() - start
    
    start = time.perf_counter()
    for node_id, labels, props in merge_nodes:
        neo4j_node = SimpleNode(node_id, props)
        await neo4j_adapter.add_node(neo4j_node)
    neo4j_time_single_merge = time.perf_counter() - start
    
    print(f"Node Ingestion Single (Merge - {half} existing, {len(new_nodes)} new): AGE={age_time_single_merge:.4f}s, Neo4j={neo4j_time_single_merge:.4f}s")
    
    edges = [(f"node_{i}", f"node_{(i+1) % batch_size}", "CONNECTS", {"weight": 1.0})
            for i in range(batch_size)]
    
    start = time.perf_counter()
    for source_id, target_id, rel_type, props in edges:
        await age_adapter.add_edge(source_id, target_id, rel_type, props)
    age_time_edge_single_new = time.perf_counter() - start
    
    start = time.perf_counter()
    for source_id, target_id, rel_type, props in edges:
        try:
            src_uuid = UUID(source_id) if '-' in source_id else UUID(int=hash(source_id) & ((1 << 128) - 1))
        except:
            src_uuid = UUID(int=hash(source_id) & ((1 << 128) - 1))
        try:
            tgt_uuid = UUID(target_id) if '-' in target_id else UUID(int=hash(target_id) & ((1 << 128) - 1))
        except:
            tgt_uuid = UUID(int=hash(target_id) & ((1 << 128) - 1))
        await neo4j_adapter.add_edge(src_uuid, tgt_uuid, rel_type, props)
    neo4j_time_edge_single_new = time.perf_counter() - start
    
    print(f"Edge Ingestion Single (New): AGE={age_time_edge_single_new:.4f}s, Neo4j={neo4j_time_edge_single_new:.4f}s")
    
    half_edges = batch_size // 2
    existing_edges = edges[:half_edges]
    new_edge_ids = [(f"node_{i}", f"node_{(i+1) % batch_size}", "CONNECTS", {"weight": 1.0})
                    for i in range(batch_size, batch_size + half_edges)]
    merge_edges = existing_edges + new_edge_ids
    
    start = time.perf_counter()
    for source_id, target_id, rel_type, props in merge_edges:
        await age_adapter.add_edge(source_id, target_id, rel_type, props)
    age_time_edge_single_merge = time.perf_counter() - start
    
    start = time.perf_counter()
    for source_id, target_id, rel_type, props in merge_edges:
        try:
            src_uuid = UUID(source_id) if '-' in source_id else UUID(int=hash(source_id) & ((1 << 128) - 1))
        except:
            src_uuid = UUID(int=hash(source_id) & ((1 << 128) - 1))
        try:
            tgt_uuid = UUID(target_id) if '-' in target_id else UUID(int=hash(target_id) & ((1 << 128) - 1))
        except:
            tgt_uuid = UUID(int=hash(target_id) & ((1 << 128) - 1))
        await neo4j_adapter.add_edge(src_uuid, tgt_uuid, rel_type, props)
    neo4j_time_edge_single_merge = time.perf_counter() - start
    
    print(f"Edge Ingestion Single (Merge - {half_edges} existing, {len(new_edge_ids)} new): AGE={age_time_edge_single_merge:.4f}s, Neo4j={neo4j_time_edge_single_merge:.4f}s")
    
    await age_adapter.drop_graph(recreate=True)
    await neo4j_adapter.delete_graph()
    
    start = time.perf_counter()
    for i in range(0, len(nodes), 100):
        await age_adapter.add_nodes(nodes[i:i+100])
    age_time_batch_new = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(0, len(nodes), 100):
        batch = nodes[i:i+100]
        neo4j_nodes = [SimpleNode(node_id, props) for node_id, _, props in batch]
        await neo4j_adapter.add_nodes(neo4j_nodes)
    neo4j_time_batch_new = time.perf_counter() - start
    
    print(f"Node Ingestion Batch (New): AGE={age_time_batch_new:.4f}s, Neo4j={neo4j_time_batch_new:.4f}s")
    
    for i in range(0, len(existing_nodes), 100):
        await age_adapter.add_nodes(existing_nodes[i:i+100])
        batch = existing_nodes[i:i+100]
        neo4j_existing = [SimpleNode(node_id, props) for node_id, _, props in batch]
        await neo4j_adapter.add_nodes(neo4j_existing)
    
    start = time.perf_counter()
    for i in range(0, len(merge_nodes), 100):
        await age_adapter.add_nodes(merge_nodes[i:i+100])
    age_time_batch_merge = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(0, len(merge_nodes), 100):
        batch = merge_nodes[i:i+100]
        neo4j_merge_nodes = [SimpleNode(node_id, props) for node_id, _, props in batch]
        await neo4j_adapter.add_nodes(neo4j_merge_nodes)
    neo4j_time_batch_merge = time.perf_counter() - start
    
    print(f"Node Ingestion Batch (Merge - {half} existing, {len(new_nodes)} new): AGE={age_time_batch_merge:.4f}s, Neo4j={neo4j_time_batch_merge:.4f}s")
    
    start = time.perf_counter()
    for i in range(0, len(edges), 100):
        await age_adapter.add_edges(edges[i:i+100])
    age_time_edge_batch_new = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(0, len(edges), 100):
        batch = edges[i:i+100]
        def to_uuid(s):
            try:
                return UUID(s) if '-' in s else UUID(int=hash(s) & ((1 << 128) - 1))
            except:
                return UUID(int=hash(s) & ((1 << 128) - 1))
        edge_tuples = [(to_uuid(src), to_uuid(tgt), rel_type, props) 
                       for src, tgt, rel_type, props in batch]
        await neo4j_adapter.add_edges(edge_tuples)
    neo4j_time_edge_batch_new = time.perf_counter() - start
    
    print(f"Edge Ingestion Batch (New): AGE={age_time_edge_batch_new:.4f}s, Neo4j={neo4j_time_edge_batch_new:.4f}s")
    
    for i in range(0, len(existing_edges), 100):
        await age_adapter.add_edges(existing_edges[i:i+100])
        batch = existing_edges[i:i+100]
        def to_uuid(s):
            try:
                return UUID(s) if '-' in s else UUID(int=hash(s) & ((1 << 128) - 1))
            except:
                return UUID(int=hash(s) & ((1 << 128) - 1))
        edge_tuples = [(to_uuid(src), to_uuid(tgt), rel_type, props) 
                       for src, tgt, rel_type, props in batch]
        await neo4j_adapter.add_edges(edge_tuples)
    
    start = time.perf_counter()
    for i in range(0, len(merge_edges), 100):
        await age_adapter.add_edges(merge_edges[i:i+100])
    age_time_edge_batch_merge = time.perf_counter() - start
    
    start = time.perf_counter()
    for i in range(0, len(merge_edges), 100):
        batch = merge_edges[i:i+100]
        def to_uuid(s):
            try:
                return UUID(s) if '-' in s else UUID(int=hash(s) & ((1 << 128) - 1))
            except:
                return UUID(int=hash(s) & ((1 << 128) - 1))
        edge_tuples = [(to_uuid(src), to_uuid(tgt), rel_type, props) 
                       for src, tgt, rel_type, props in batch]
        await neo4j_adapter.add_edges(edge_tuples)
    neo4j_time_edge_batch_merge = time.perf_counter() - start
    
    print(f"Edge Ingestion Batch (Merge - {half_edges} existing, {len(new_edge_ids)} new): AGE={age_time_edge_batch_merge:.4f}s, Neo4j={neo4j_time_edge_batch_merge:.4f}s")
    
    query_node_ids = [f"node_{i}" for i in range(0, batch_size, 10)]
    
    start = time.perf_counter()
    for node_id in query_node_ids:
        await age_adapter.get_node(node_id)
    age_time_get_node = time.perf_counter() - start
    
    start = time.perf_counter()
    for node_id in query_node_ids:
        try:
            node_uuid = UUID(node_id) if '-' in node_id else UUID(int=hash(node_id) & ((1 << 128) - 1))
        except:
            node_uuid = UUID(int=hash(node_id) & ((1 << 128) - 1))
        await neo4j_adapter.get_node(str(node_uuid))
    neo4j_time_get_node = time.perf_counter() - start
    
    print(f"Get Node by ID ({len(query_node_ids)} queries): AGE={age_time_get_node:.4f}s, Neo4j={neo4j_time_get_node:.4f}s")
    
    start = time.perf_counter()
    for node_id in query_node_ids[:10]:
        await age_adapter.get_neighbors(node_id)
    age_time_get_neighbors = time.perf_counter() - start
    
    start = time.perf_counter()
    for node_id in query_node_ids[:10]:
        try:
            node_uuid = UUID(node_id) if '-' in node_id else UUID(int=hash(node_id) & ((1 << 128) - 1))
        except:
            node_uuid = UUID(int=hash(node_id) & ((1 << 128) - 1))
        query = f"""
        MATCH (n:`__Node__`{{id: $node_id}})-[]-(neighbor:`__Node__`)
        RETURN DISTINCT neighbor.id as id, properties(neighbor) as properties
        """
        await neo4j_adapter.query(query, {"node_id": str(node_uuid)})
    neo4j_time_get_neighbors = time.perf_counter() - start
    
    print(f"Get Neighbors ({len(query_node_ids[:10])} queries): AGE={age_time_get_neighbors:.4f}s, Neo4j={neo4j_time_get_neighbors:.4f}s")
    
    start = time.perf_counter()
    for node_id in query_node_ids[:10]:
        await age_adapter.get_edges(node_id)
    age_time_get_edges = time.perf_counter() - start
    
    start = time.perf_counter()
    for node_id in query_node_ids[:10]:
        try:
            node_uuid = UUID(node_id) if '-' in node_id else UUID(int=hash(node_id) & ((1 << 128) - 1))
        except:
            node_uuid = UUID(int=hash(node_id) & ((1 << 128) - 1))
        await neo4j_adapter.get_edges(str(node_uuid))
    neo4j_time_get_edges = time.perf_counter() - start
    
    print(f"Get Edges ({len(query_node_ids[:10])} queries): AGE={age_time_get_edges:.4f}s, Neo4j={neo4j_time_get_edges:.4f}s")
    
    start = time.perf_counter()
    query = "MATCH (n:TestNode) WHERE n.value > 1000 RETURN n LIMIT 100"
    await age_adapter.execute_cypher(query)
    age_time_prop_filter = time.perf_counter() - start
    
    start = time.perf_counter()
    query = f"""
    MATCH (n:`__Node__`)
    WHERE n.value > 1000
    RETURN n
    LIMIT 100
    """
    await neo4j_adapter.query(query)
    neo4j_time_prop_filter = time.perf_counter() - start
    
    print(f"Property Filter (value > 1000, limit 100): AGE={age_time_prop_filter:.4f}s, Neo4j={neo4j_time_prop_filter:.4f}s")
    
    start = time.perf_counter()
    query = "MATCH (n) WHERE n.name CONTAINS 'Node 1' RETURN n LIMIT 100"
    await age_adapter.execute_cypher(query)
    age_time_text_search = time.perf_counter() - start
    
    start = time.perf_counter()
    query = f"""
    MATCH (n:`__Node__`)
    WHERE n.name CONTAINS 'Node 1'
    RETURN n
    LIMIT 100
    """
    await neo4j_adapter.query(query)
    neo4j_time_text_search = time.perf_counter() - start
    
    print(f"Text Search (name CONTAINS 'Node 1', limit 100): AGE={age_time_text_search:.4f}s, Neo4j={neo4j_time_text_search:.4f}s")
    
    start = time.perf_counter()
    query = "MATCH (n) RETURN {count: count(n)}"
    await age_adapter.execute_cypher(query)
    age_time_count = time.perf_counter() - start
    
    start = time.perf_counter()
    query = f"MATCH (n:`__Node__`) RETURN count(n) as count"
    await neo4j_adapter.query(query)
    neo4j_time_count = time.perf_counter() - start
    
    print(f"Count Nodes: AGE={age_time_count:.4f}s, Neo4j={neo4j_time_count:.4f}s")
    
    start = time.perf_counter()
    query = "MATCH ()-[r]->() RETURN {count: count(r)}"
    await age_adapter.execute_cypher(query)
    age_time_count_edges = time.perf_counter() - start
    
    start = time.perf_counter()
    query = "MATCH ()-[r]->() RETURN count(r) as count"
    await neo4j_adapter.query(query)
    neo4j_time_count_edges = time.perf_counter() - start
    
    print(f"Count Edges: AGE={age_time_count_edges:.4f}s, Neo4j={neo4j_time_count_edges:.4f}s")
    
    
    await age_adapter.close()
    await neo4j_adapter.driver.close()


if __name__ == "__main__":
    asyncio.run(main())

