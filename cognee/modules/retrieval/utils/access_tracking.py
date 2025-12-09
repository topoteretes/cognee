"""Utilities for tracking data access in retrievers."""  
  
import json  
from datetime import datetime, timezone  
from typing import List, Any  
from uuid import UUID  
import os   
from cognee.infrastructure.databases.graph import get_graph_engine  
from cognee.infrastructure.databases.relational import get_relational_engine  
from cognee.modules.data.models import Data  
from cognee.shared.logging_utils import get_logger  
from sqlalchemy import update  
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph  
  
logger = get_logger(__name__)  
  
async def update_node_access_timestamps(items: List[Any]):  
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":  
        return  
          
    if not items:  
        return  
          
    graph_engine = await get_graph_engine()  
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  
    timestamp_dt = datetime.now(timezone.utc)  
          
    # Extract node IDs  
    node_ids = []  
    for item in items:  
        item_id = item.payload.get("id") if hasattr(item, 'payload') else item.get("id")  
        if item_id:  
            node_ids.append(str(item_id))  
          
    if not node_ids:  
        return  
          
    try:  
        # Try to update nodes in graph database (may fail for unsupported DBs)  
        await _update_nodes_via_projection(graph_engine, node_ids, timestamp_ms)  
    except Exception as e:  
        logger.warning(  
            f"Failed to update node timestamps in graph database: {e}. "  
            "Will update document-level timestamps in SQL instead."  
        )  
              
    # Always try to find origin documents and update SQL  
    # This ensures document-level tracking works even if graph updates fail  
    try:  
        doc_ids = await _find_origin_documents_via_projection(graph_engine, node_ids)  
        if doc_ids:  
            await _update_sql_records(doc_ids, timestamp_dt)  
    except Exception as e:  
        logger.error(f"Failed to update SQL timestamps: {e}")  
        raise  
  
async def _update_nodes_via_projection(graph_engine, node_ids, timestamp_ms):  
    """Update nodes using graph projection - works with any graph database"""  
    # Project the graph with necessary properties  
    memory_fragment = CogneeGraph()  
    await memory_fragment.project_graph_from_db(  
        graph_engine,  
        node_properties_to_project=["id"],  
        edge_properties_to_project=[]  
    )  
      
    # Update each node's last_accessed_at property  
    provider = os.getenv("GRAPH_DATABASE_PROVIDER", "kuzu").lower()  
      
    for node_id in node_ids:  
        node = memory_fragment.get_node(node_id)  
        if node:  
            try:  
                # Update the node in the database  
                if provider == "kuzu":  
                    # Kuzu stores properties as JSON  
                    result = await graph_engine.query(  
                        "MATCH (n:Node {id: $id}) RETURN n.properties",  
                        {"id": node_id}  
                    )  
                      
                    if result and result[0]:  
                        props = json.loads(result[0][0]) if result[0][0] else {}  
                        props["last_accessed_at"] = timestamp_ms  
                          
                        await graph_engine.query(  
                            "MATCH (n:Node {id: $id}) SET n.properties = $props",  
                            {"id": node_id, "props": json.dumps(props)}  
                        )  
                elif provider == "neo4j":  
                    await graph_engine.query(  
                        "MATCH (n:__Node__ {id: $id}) SET n.last_accessed_at = $timestamp",  
                        {"id": node_id, "timestamp": timestamp_ms}  
                    )  
                elif provider == "neptune":  
                    await graph_engine.query(  
                        "MATCH (n:Node {id: $id}) SET n.last_accessed_at = $timestamp",  
                        {"id": node_id, "timestamp": timestamp_ms}  
                    )  
            except Exception as e:  
                # Log but continue with other nodes  
                logger.debug(f"Failed to update node {node_id}: {e}")  
                continue  
  
async def _find_origin_documents_via_projection(graph_engine, node_ids):  
    """Find origin documents using graph projection instead of DB queries"""  
    # Project the entire graph with necessary properties  
    memory_fragment = CogneeGraph()  
    await memory_fragment.project_graph_from_db(  
        graph_engine,  
        node_properties_to_project=["id", "type"],  
        edge_properties_to_project=["relationship_name"]  
    )  
      
    # Find origin documents by traversing the in-memory graph  
    doc_ids = set()  
    for node_id in node_ids:  
        node = memory_fragment.get_node(node_id)  
        if node and node.get_attribute("type") == "DocumentChunk":  
            # Traverse edges to find connected documents  
            for edge in node.get_skeleton_edges():  
                # Get the neighbor node  
                neighbor = edge.get_destination_node() if edge.get_source_node().id == node_id else edge.get_source_node()  
                if neighbor and neighbor.get_attribute("type") in ["TextDocument", "Document"]:  
                    doc_ids.add(neighbor.id)  
      
    return list(doc_ids)  
  
async def _update_sql_records(doc_ids, timestamp_dt):  
    """Update SQL Data table (same for all providers)"""  
    db_engine = get_relational_engine()  
    async with db_engine.get_async_session() as session:  
        stmt = update(Data).where(  
            Data.id.in_([UUID(doc_id) for doc_id in doc_ids])  
        ).values(last_accessed=timestamp_dt)  
              
        await session.execute(stmt)  
        await session.commit()
