"""Utilities for tracking data access in retrievers."""  
  
import json  
from datetime import datetime, timezone  
from typing import List, Any  
from uuid import UUID  
  
from cognee.infrastructure.databases.graph import get_graph_engine  
from cognee.infrastructure.databases.relational import get_relational_engine  
from cognee.modules.data.models import Data  
from cognee.shared.logging_utils import get_logger  
from sqlalchemy import update  
  
logger = get_logger(__name__)  
  
  
async def update_node_access_timestamps(items: List[Any]):  
    """  
    Update last_accessed_at for nodes in graph database and corresponding Data records in SQL.  
      
    This function:  
    1. Updates last_accessed_at in the graph database nodes (in properties JSON)  
    2. Traverses to find origin TextDocument nodes (without hardcoded relationship names)  
    3. Updates last_accessed in the SQL Data table for those documents  
      
    Parameters  
    ----------  
    items : List[Any]  
        List of items with payload containing 'id' field (from vector search results)  
    """  
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
        # Step 1: Batch update graph nodes  
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
          
        logger.debug(f"Updated access timestamps for {len(node_ids)} graph nodes")  
          
        # Step 2: Find origin TextDocument nodes (without hardcoded relationship names)  
        origin_query = """  
        UNWIND $node_ids AS node_id  
        MATCH (chunk:Node {id: node_id})-[e:EDGE]-(doc:Node)  
        WHERE chunk.type = 'DocumentChunk' AND doc.type IN ['TextDocument', 'Document']  
        RETURN DISTINCT doc.id  
        """  
          
        result = await graph_engine.query(origin_query, {"node_ids": node_ids})  
          
        # Extract and deduplicate document IDs  
        doc_ids = list(set([row[0] for row in result if row and row[0]])) if result else []  
          
        # Step 3: Update SQL Data table  
        if doc_ids:  
            db_engine = get_relational_engine()  
            async with db_engine.get_async_session() as session:  
                stmt = update(Data).where(  
                    Data.id.in_([UUID(doc_id) for doc_id in doc_ids])  
                ).values(last_accessed=timestamp_dt)  
                  
                await session.execute(stmt)  
                await session.commit()  
                  
            logger.debug(f"Updated last_accessed for {len(doc_ids)} Data records in SQL")  
          
    except Exception as e:  
        logger.error(f"Failed to update timestamps: {e}")  
        raise
