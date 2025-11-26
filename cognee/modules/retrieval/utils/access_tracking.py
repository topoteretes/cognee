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
        # Detect database provider and use appropriate queries  
        provider = os.getenv("GRAPH_DATABASE_PROVIDER", "kuzu").lower()  
          
        if provider == "kuzu":  
            await _update_kuzu_nodes(graph_engine, node_ids, timestamp_ms)  
        elif provider == "neo4j":  
            await _update_neo4j_nodes(graph_engine, node_ids, timestamp_ms)  
        elif provider == "neptune":  
            await _update_neptune_nodes(graph_engine, node_ids, timestamp_ms)  
        else:  
            logger.warning(f"Unsupported graph provider: {provider}")  
            return  
              
        # Find origin documents and update SQL  
        doc_ids = await _find_origin_documents(graph_engine, node_ids, provider)  
        if doc_ids:  
            await _update_sql_records(doc_ids, timestamp_dt)  
              
    except Exception as e:  
        logger.error(f"Failed to update timestamps: {e}")  
        raise  
  
async def _update_kuzu_nodes(graph_engine, node_ids, timestamp_ms):  
    """Kuzu-specific node updates"""  
    for node_id in node_ids:  
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
  
async def _update_neo4j_nodes(graph_engine, node_ids, timestamp_ms):  
    """Neo4j-specific node updates"""  
    for node_id in node_ids:  
        await graph_engine.query(  
            "MATCH (n:__Node__ {id: $id}) SET n.last_accessed_at = $timestamp",  
            {"id": node_id, "timestamp": timestamp_ms}  
        )  
  
async def _update_neptune_nodes(graph_engine, node_ids, timestamp_ms):  
    """Neptune-specific node updates"""  
    for node_id in node_ids:  
        await graph_engine.query(  
            "MATCH (n:Node {id: $id}) SET n.last_accessed_at = $timestamp",  
            {"id": node_id, "timestamp": timestamp_ms}  
        )  
  
async def _find_origin_documents(graph_engine, node_ids, provider):  
    """Find origin documents with provider-specific queries"""  
    if provider == "kuzu":  
        query = """  
        UNWIND $node_ids AS node_id    
        MATCH (chunk:Node {id: node_id})-[e:EDGE]-(doc:Node)    
        WHERE chunk.type = 'DocumentChunk' AND doc.type IN ['TextDocument', 'Document']    
        RETURN DISTINCT doc.id  
        """  
    elif provider == "neo4j":  
        query = """  
        UNWIND $node_ids AS node_id    
        MATCH (chunk:__Node__ {id: node_id})-[e:EDGE]-(doc:__Node__)    
        WHERE chunk.type = 'DocumentChunk' AND doc.type IN ['TextDocument', 'Document']    
        RETURN DISTINCT doc.id  
        """  
    elif provider == "neptune":  
        query = """  
        UNWIND $node_ids AS node_id    
        MATCH (chunk:Node {id: node_id})-[e:EDGE]-(doc:Node)    
        WHERE chunk.type = 'DocumentChunk' AND doc.type IN ['TextDocument', 'Document']    
        RETURN DISTINCT doc.id  
        """  
      
    result = await graph_engine.query(query, {"node_ids": node_ids})  
    return list(set([row[0] for row in result if row and row[0]])) if result else []  
  
async def _update_sql_records(doc_ids, timestamp_dt):  
    """Update SQL Data table (same for all providers)"""  
    db_engine = get_relational_engine()  
    async with db_engine.get_async_session() as session:  
        stmt = update(Data).where(  
            Data.id.in_([UUID(doc_id) for doc_id in doc_ids])  
        ).values(last_accessed=timestamp_dt)  
          
        await session.execute(stmt)  
        await session.commit()
