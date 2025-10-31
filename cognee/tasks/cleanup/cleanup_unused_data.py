"""  
Task for automatically deleting unused data from the memify pipeline.  
  
This task identifies and removes data (chunks, entities, summaries) that hasn't  
been accessed by retrievers for a specified period, helping maintain system  
efficiency and storage optimization.  
"""  
  
import json  
from datetime import datetime, timezone, timedelta  
from typing import Optional, Dict, Any  
from uuid import UUID  
  
from cognee.infrastructure.databases.graph import get_graph_engine  
from cognee.infrastructure.databases.vector import get_vector_engine  
from cognee.shared.logging_utils import get_logger  
  
logger = get_logger(__name__)  
  
  
async def cleanup_unused_data(  
    minutes_threshold: int = 30,  
    dry_run: bool = True,  
    user_id: Optional[UUID] = None  
) -> Dict[str, Any]:  
    """  
    Identify and remove unused data from the memify pipeline.  
      
    Parameters  
    ----------  
    minutes_threshold : int  
        Minutes since last access to consider data unused (default: 30)  
    dry_run : bool  
        If True, only report what would be deleted without actually deleting (default: True)  
    user_id : UUID, optional  
        Limit cleanup to specific user's data (default: None)  
      
    Returns  
    -------  
    Dict[str, Any]  
        Cleanup results with status, counts, and timestamp  
    """  
    logger.info(  
        "Starting cleanup task",  
        minutes_threshold=minutes_threshold,  
        dry_run=dry_run,  
        user_id=str(user_id) if user_id else None  
    )  
      
    # Calculate cutoff timestamp in milliseconds  
    cutoff_date = datetime.now(timezone.utc) - timedelta(minutes=minutes_threshold)  
    cutoff_timestamp_ms = int(cutoff_date.timestamp() * 1000)  
      
    logger.debug(f"Cutoff timestamp: {cutoff_date.isoformat()} ({cutoff_timestamp_ms}ms)")  
      
    # Find unused nodes  
    unused_nodes = await _find_unused_nodes(cutoff_timestamp_ms, user_id)  
      
    total_unused = sum(len(nodes) for nodes in unused_nodes.values())  
    logger.info(f"Found {total_unused} unused nodes", unused_nodes={k: len(v) for k, v in unused_nodes.items()})  
      
    if dry_run:  
        return {  
            "status": "dry_run",  
            "unused_count": total_unused,  
            "deleted_count": {  
                "data_items": 0,  
                "chunks": 0,  
                "entities": 0,  
                "summaries": 0,  
                "associations": 0  
            },  
            "cleanup_date": datetime.now(timezone.utc).isoformat(),  
            "preview": {  
                "chunks": len(unused_nodes["DocumentChunk"]),  
                "entities": len(unused_nodes["Entity"]),  
                "summaries": len(unused_nodes["TextSummary"])  
            }  
        }  
      
    # Delete unused nodes  
    deleted_counts = await _delete_unused_nodes(unused_nodes)  
      
    logger.info("Cleanup completed", deleted_counts=deleted_counts)  
      
    return {  
        "status": "completed",  
        "unused_count": total_unused,  
        "deleted_count": {  
            "data_items": 0,  
            "chunks": deleted_counts["DocumentChunk"],  
            "entities": deleted_counts["Entity"],  
            "summaries": deleted_counts["TextSummary"],  
            "associations": deleted_counts["associations"]  
        },  
        "cleanup_date": datetime.now(timezone.utc).isoformat()  
    }  
  
  
async def _find_unused_nodes(  
    cutoff_timestamp_ms: int,  
    user_id: Optional[UUID] = None  
) -> Dict[str, list]:  
    """  
    Query Kuzu for nodes with old last_accessed_at timestamps.  
      
    Parameters  
    ----------  
    cutoff_timestamp_ms : int  
        Cutoff timestamp in milliseconds since epoch  
    user_id : UUID, optional  
        Filter by user ID if provided  
      
    Returns  
    -------  
    Dict[str, list]  
        Dictionary mapping node types to lists of unused node IDs  
    """  
    graph_engine = await get_graph_engine()  
      
    # Query all nodes with their properties  
    query = "MATCH (n:Node) RETURN n.id, n.type, n.properties"  
    results = await graph_engine.query(query)  
      
    unused_nodes = {  
        "DocumentChunk": [],  
        "Entity": [],  
        "TextSummary": []  
    }  
      
    for node_id, node_type, props_json in results:  
        # Only process tracked node types  
        if node_type not in unused_nodes:  
            continue  
          
        # Parse properties JSON  
        if props_json:  
            try:  
                props = json.loads(props_json)  
                last_accessed = props.get("last_accessed_at")  
                  
                # Check if node is unused (never accessed or accessed before cutoff)  
                if last_accessed is None or last_accessed < cutoff_timestamp_ms:  
                    # TODO: Add user_id filtering when user ownership is implemented  
                    unused_nodes[node_type].append(node_id)  
                    logger.debug(  
                        f"Found unused {node_type}",  
                        node_id=node_id,  
                        last_accessed=last_accessed  
                    )  
            except json.JSONDecodeError:  
                logger.warning(f"Failed to parse properties for node {node_id}")  
                continue  
      
    return unused_nodes  
  
  
async def _delete_unused_nodes(unused_nodes: Dict[str, list]) -> Dict[str, int]:  
    """  
    Delete unused nodes from graph and vector databases.  
      
    Parameters  
    ----------  
    unused_nodes : Dict[str, list]  
        Dictionary mapping node types to lists of node IDs to delete  
      
    Returns  
    -------  
    Dict[str, int]  
        Count of deleted items by type  
    """  
    graph_engine = await get_graph_engine()  
    vector_engine = get_vector_engine()  
      
    deleted_counts = {  
        "DocumentChunk": 0,  
        "Entity": 0,  
        "TextSummary": 0,  
        "associations": 0  
    }  
      
    # Count associations before deletion  
    for node_type, node_ids in unused_nodes.items():  
        if not node_ids:  
            continue  
          
        # Count edges connected to these nodes  
        for node_id in node_ids:  
            result = await graph_engine.query(  
                "MATCH (n:Node {id: $id})-[r:EDGE]-() RETURN count(r)",  
                {"id": node_id}  
            )  
            if result and len(result) > 0:  
                deleted_counts["associations"] += result[0][0]  
      
    # Delete from graph database (uses DETACH DELETE, so edges are automatically removed)  
    for node_type, node_ids in unused_nodes.items():  
        if not node_ids:  
            continue  
          
        logger.info(f"Deleting {len(node_ids)} {node_type} nodes from graph database")  
          
        # Delete nodes in batches  
        await graph_engine.delete_nodes(node_ids)  
        deleted_counts[node_type] = len(node_ids)  
      
    # Delete from vector database  
    vector_collections = {  
        "DocumentChunk": "DocumentChunk_text",  
        "Entity": "Entity_name",  
        "TextSummary": "TextSummary_text"  
    }  
      
    for node_type, collection_name in vector_collections.items():  
        node_ids = unused_nodes[node_type]  
        if not node_ids:  
            continue  
          
        logger.info(f"Deleting {len(node_ids)} {node_type} embeddings from vector database")  
          
        try:  
            # Delete from vector collection  
            if await vector_engine.has_collection(collection_name):  
                for node_id in node_ids:  
                    try:  
                        await vector_engine.delete(collection_name, {"id": str(node_id)})  
                    except Exception as e:  
                        logger.warning(f"Failed to delete {node_id} from {collection_name}: {e}")  
        except Exception as e:  
            logger.error(f"Error deleting from vector collection {collection_name}: {e}")  
      
    return deleted_counts
