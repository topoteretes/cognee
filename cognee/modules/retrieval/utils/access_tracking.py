
"""Utilities for tracking data access in retrievers."""  
  
import json  
from datetime import datetime, timezone  
from typing import List, Any  
  
from cognee.infrastructure.databases.graph import get_graph_engine  
from cognee.shared.logging_utils import get_logger  
  
logger = get_logger(__name__)  
  
  
async def update_node_access_timestamps(items: List[Any], node_type: str):  
    """  
    Update last_accessed_at for nodes in Kuzu graph database.  
      
    Parameters  
    ----------  
    items : List[Any]  
        List of items with payload containing 'id' field (from vector search results)  
    node_type : str  
        Type of node to update (e.g., 'DocumentChunk', 'Entity', 'TextSummary')  
    """  
    if not items:  
        return  
      
    graph_engine = await get_graph_engine()  
    # Convert to milliseconds since epoch (matching the field format)  
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)  
      
    for item in items:  
        # Extract ID from payload (vector search results have this structure)  
        item_id = item.payload.get("id") if hasattr(item, 'payload') else item.get("id")  
        if not item_id:  
            continue  
              
        try:  
            # Get current node properties from Kuzu's Node table  
            result = await graph_engine.query(  
                "MATCH (n:Node {id: $id}) WHERE n.type = $node_type RETURN n.properties as props",  
                {"id": str(item_id), "node_type": node_type}  
            )  
              
            if result and len(result) > 0 and result[0][0]:  
                # Parse existing properties JSON  
                props = json.loads(result[0][0]) if result[0][0] else {}  
                # Update last_accessed_at with millisecond timestamp  
                props["last_accessed_at"] = timestamp_ms  
                  
                # Write back to graph database  
                await graph_engine.query(  
                    "MATCH (n:Node {id: $id}) WHERE n.type = $node_type SET n.properties = $props",  
                    {"id": str(item_id), "node_type": node_type, "props": json.dumps(props)}  
                )  
        except Exception as e:  
            logger.warning(f"Failed to update timestamp for {node_type} {item_id}: {e}")  
            continue  
      
    logger.debug(f"Updated access timestamps for {len(items)} {node_type} nodes")

