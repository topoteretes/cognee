  
"""Utilities for tracking data access in retrievers."""  
  
import json  
from datetime import datetime, timezone  
from typing import List, Any  
  
from cognee.infrastructure.databases.graph import get_graph_engine  
from cognee.shared.logging_utils import get_logger  
  
logger = get_logger(__name__)  
  
  
async def update_node_access_timestamps(items: List[Any]):  
    """  
    Update last_accessed_at for nodes in Kuzu graph database.  
    Automatically determines node type from the graph database.  
      
    Parameters  
    ----------  
    items : List[Any]  
        List of items with payload containing 'id' field (from vector search results)  
    """  
    if not items:  
        return  
      
    graph_engine = await get_graph_engine()  
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  
      
    for item in items:  
        # Extract ID from payload  
        item_id = item.payload.get("id") if hasattr(item, 'payload') else item.get("id")  
        if not item_id:  
            continue  
              
        # try:  
        # Query to get both node type and properties in one call  
        result = await graph_engine.query(  
            "MATCH (n:Node {id: $id}) RETURN n.type as node_type, n.properties as props",  
            {"id": str(item_id)}  
        )  
          
        if result and len(result) > 0 and result[0]:  
            node_type = result[0][0]  # First column: node_type  
            props_json = result[0][1]  # Second column: properties  
              
            # Parse existing properties JSON  
            props = json.loads(props_json) if props_json else {}  
            # Update last_accessed_at with millisecond timestamp  
            props["last_accessed_at"] = timestamp_ms  
              
            # Write back to graph database  
            await graph_engine.query(  
                "MATCH (n:Node {id: $id}) SET n.properties = $props",  
                {"id": str(item_id), "props": json.dumps(props)}  
            )  
              
            logger.debug(f"Updated access timestamp for {node_type} node {item_id}")  
                  
        # except Exception as e:  
        #     logger.error(f"Failed to update timestamp for node {item_id}: {e}")  
        #     continue  
      
    logger.debug(f"Updated access timestamps for {len(items)} nodes")
