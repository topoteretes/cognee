"""        
Task for automatically deleting unused data from the memify pipeline.        
        
This task identifies and removes entire documents that haven't        
been accessed by retrievers for a specified period, helping maintain system        
efficiency and storage optimization through whole-document removal.        
"""        
        
import json        
from datetime import datetime, timezone, timedelta        
from typing import Optional, Dict, Any        
from uuid import UUID        
import os        
from cognee.infrastructure.databases.graph import get_graph_engine        
from cognee.infrastructure.databases.vector import get_vector_engine        
from cognee.infrastructure.databases.relational import get_relational_engine      
from cognee.modules.data.models import Data, DatasetData      
from cognee.shared.logging_utils import get_logger        
from sqlalchemy import select, or_      
import cognee      
import sqlalchemy as sa    
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph  
        
logger = get_logger(__name__)        
        
        
async def cleanup_unused_data(        
    minutes_threshold: Optional[int],        
    dry_run: bool = True,        
    user_id: Optional[UUID] = None,      
    text_doc: bool = True,  # Changed default to True for document-level cleanup  
    node_level: bool = False  # New parameter for explicit node-level cleanup  
) -> Dict[str, Any]:        
    """        
    Identify and remove unused data from the memify pipeline.        
        
    Parameters        
    ----------        
    minutes_threshold : int        
        Minutes since last access to consider data unused         
    dry_run : bool        
        If True, only report what would be deleted without actually deleting (default: True)        
    user_id : UUID, optional        
        Limit cleanup to specific user's data (default: None)      
    text_doc : bool      
        If True (default), use SQL-based filtering to find unused TextDocuments and call cognee.delete()      
        for proper whole-document deletion      
    node_level : bool      
        If True, perform chaotic node-level deletion of unused chunks, entities, and summaries      
        (default: False - deprecated in favor of document-level cleanup)      
            
    Returns        
    -------        
    Dict[str, Any]        
        Cleanup results with status, counts, and timestamp        
    """       
    # Check 1: Environment variable must be enabled      
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":      
        logger.warning(      
            "Cleanup skipped: ENABLE_LAST_ACCESSED is not enabled."      
        )      
        return {      
            "status": "skipped",      
            "reason": "ENABLE_LAST_ACCESSED not enabled",      
            "unused_count": 0,      
            "deleted_count": {},      
            "cleanup_date": datetime.now(timezone.utc).isoformat()      
        }      
          
    # Check 2: Verify tracking has actually been running      
    db_engine = get_relational_engine()      
    async with db_engine.get_async_session() as session:      
        # Count records with non-NULL last_accessed      
        tracked_count = await session.execute(      
            select(sa.func.count(Data.id)).where(Data.last_accessed.isnot(None))      
        )      
        tracked_records = tracked_count.scalar()      
              
        if tracked_records == 0:      
            logger.warning(      
                "Cleanup skipped: No records have been tracked yet. "      
                "ENABLE_LAST_ACCESSED may have been recently enabled. "      
                "Wait for retrievers to update timestamps before running cleanup."      
            )      
            return {      
                "status": "skipped",      
                "reason": "No tracked records found - tracking may be newly enabled",      
                "unused_count": 0,      
                "deleted_count": {},      
                "cleanup_date": datetime.now(timezone.utc).isoformat()      
            }      
          
    logger.info(        
        "Starting cleanup task",        
        minutes_threshold=minutes_threshold,        
        dry_run=dry_run,        
        user_id=str(user_id) if user_id else None,      
        text_doc=text_doc,      
        node_level=node_level      
    )        
            
    # Calculate cutoff timestamp      
    cutoff_date = datetime.now(timezone.utc) - timedelta(minutes=minutes_threshold)    
          
    if node_level:      
        # Deprecated: Node-level approach (chaotic)      
        logger.warning(      
            "Node-level cleanup is deprecated and may lead to fragmented knowledge graphs. "      
            "Consider using document-level cleanup (default) instead."      
        )      
        cutoff_timestamp_ms = int(cutoff_date.timestamp() * 1000)      
        logger.debug(f"Cutoff timestamp: {cutoff_date.isoformat()} ({cutoff_timestamp_ms}ms)")        
                
        # Find unused nodes using graph projection    
        unused_nodes = await _find_unused_nodes_via_projection(cutoff_timestamp_ms)    
                
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
                
        # Delete unused nodes (provider-agnostic deletion)    
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
    else:      
        # Default: Document-level approach (recommended)      
        return await _cleanup_via_sql(cutoff_date, dry_run, user_id)      
      
      
async def _cleanup_via_sql(      
    cutoff_date: datetime,      
    dry_run: bool,      
    user_id: Optional[UUID] = None      
) -> Dict[str, Any]:      
    """      
    SQL-based cleanup: Query Data table for unused documents and use cognee.delete().      
          
    Parameters      
    ----------      
    cutoff_date : datetime      
        Cutoff date for last_accessed filtering      
    dry_run : bool      
        If True, only report what would be deleted      
    user_id : UUID, optional      
        Filter by user ID if provided      
          
    Returns      
    -------      
    Dict[str, Any]      
        Cleanup results      
    """      
    db_engine = get_relational_engine()      
          
    async with db_engine.get_async_session() as session:      
        # Query for Data records with old last_accessed timestamps      
        query = select(Data, DatasetData).join(      
            DatasetData, Data.id == DatasetData.data_id      
        ).where(      
            or_(      
                Data.last_accessed < cutoff_date,      
                Data.last_accessed.is_(None)      
            )      
        )      
              
        if user_id:      
            from cognee.modules.data.models import Dataset      
            query = query.join(Dataset, DatasetData.dataset_id == Dataset.id).where(      
                Dataset.owner_id == user_id      
            )      
              
        result = await session.execute(query)      
        unused_data = result.all()      
          
    logger.info(f"Found {len(unused_data)} unused documents in SQL")      
          
    if dry_run:      
        return {      
            "status": "dry_run",      
            "unused_count": len(unused_data),      
            "deleted_count": {      
                "data_items": 0,      
                "documents": 0      
            },      
            "cleanup_date": datetime.now(timezone.utc).isoformat(),      
            "preview": {      
                "documents": len(unused_data)      
            }      
        }      
          
    # Delete each document using cognee.delete()      
    deleted_count = 0      
    from cognee.modules.users.methods import get_default_user      
    user = await get_default_user() if user_id is None else None      
          
    for data, dataset_data in unused_data:      
        try:      
            await cognee.delete(      
                data_id=data.id,      
                dataset_id=dataset_data.dataset_id,      
                mode="hard",  # Use hard mode to also remove orphaned entities      
                user=user      
            )      
            deleted_count += 1      
            logger.info(f"Deleted document {data.id} from dataset {dataset_data.dataset_id}")      
        except Exception as e:      
            logger.error(f"Failed to delete document {data.id}: {e}")      
          
    logger.info("Cleanup completed", deleted_count=deleted_count)      
          
    return {      
        "status": "completed",      
        "unused_count": len(unused_data),      
        "deleted_count": {      
            "data_items": deleted_count,      
            "documents": deleted_count      
        },      
        "cleanup_date": datetime.now(timezone.utc).isoformat()      
    }      
        
        
async def _find_unused_nodes_via_projection(cutoff_timestamp_ms: int) -> Dict[str, list]:        
    """        
    Find unused nodes using graph projection - database-agnostic approach.        
    NOTE: This function is deprecated as it leads to fragmented knowledge graphs.        
            
    Parameters        
    ----------        
    cutoff_timestamp_ms : int        
        Cutoff timestamp in milliseconds since epoch        
            
    Returns        
    -------        
    Dict[str, list]        
        Dictionary mapping node types to lists of unused node IDs        
    """        
    graph_engine = await get_graph_engine()        
            
    # Project the entire graph with necessary properties    
    memory_fragment = CogneeGraph()    
    await memory_fragment.project_graph_from_db(    
        graph_engine,    
        node_properties_to_project=["id", "type", "last_accessed_at"],    
        edge_properties_to_project=[]    
    )    
        
    unused_nodes = {"DocumentChunk": [], "Entity": [], "TextSummary": []}    
        
    # Get all nodes from the projected graph    
    all_nodes = memory_fragment.get_nodes()    
        
    for node in all_nodes:    
        node_type = node.get_attribute("type")    
        if node_type not in unused_nodes:    
            continue    
                
        # Check last_accessed_at property    
        last_accessed = node.get_attribute("last_accessed_at")    
            
        if last_accessed is None or last_accessed < cutoff_timestamp_ms:    
            unused_nodes[node_type].append(node.id)    
            logger.debug(    
                f"Found unused {node_type}",    
                node_id=node.id,    
                last_accessed=last_accessed    
            )    
        
    return unused_nodes    
    
    
async def _delete_unused_nodes(unused_nodes: Dict[str, list]) -> Dict[str, int]:        
    """        
    Delete unused nodes from graph and vector databases.        
    NOTE: This function is deprecated as it leads to fragmented knowledge graphs.        
            
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
            
    # Count associations before deletion (using graph projection for consistency)    
    if any(unused_nodes.values()):    
        memory_fragment = CogneeGraph()    
        await memory_fragment.project_graph_from_db(    
            graph_engine,    
            node_properties_to_project=["id"],    
            edge_properties_to_project=[]    
        )    
            
        for node_type, node_ids in unused_nodes.items():        
            if not node_ids:        
                continue        
                    
            # Count edges from the in-memory graph    
            for node_id in node_ids:    
                node = memory_fragment.get_node(node_id)    
                if node:    
                    # Count edges from the in-memory graph    
                    edge_count = len(node.get_skeleton_edges())    
                    deleted_counts["associations"] += edge_count    
        
    # Delete from graph database (uses DETACH DELETE, so edges are automatically removed)        
    for node_type, node_ids in unused_nodes.items():        
        if not node_ids:        
            continue        
                
        logger.info(f"Deleting {len(node_ids)} {node_type} nodes from graph database")        
                
        # Delete nodes in batches (database-agnostic)        
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
            if await vector_engine.has_collection(collection_name):    
                await vector_engine.delete_data_points(    
                    collection_name,    
                    [str(node_id) for node_id in node_ids]    
                )    
        except Exception as e:    
            logger.error(f"Error deleting from vector collection {collection_name}: {e}")    
            
    return deleted_counts
