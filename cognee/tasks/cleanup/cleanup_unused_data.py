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
from cognee.infrastructure.databases.relational import get_relational_engine  
from cognee.modules.data.models import Data, DatasetData  
from cognee.shared.logging_utils import get_logger    
from sqlalchemy import select, or_  
import cognee  
    
logger = get_logger(__name__)    
    
    
async def cleanup_unused_data(    
    days_threshold: Optional[int],    
    dry_run: bool = True,    
    user_id: Optional[UUID] = None,  
    text_doc: bool = False  
) -> Dict[str, Any]:    
    """    
    Identify and remove unused data from the memify pipeline.    
        
    Parameters    
    ----------    
    days_threshold : int    
        days since last access to consider data unused     
    dry_run : bool    
        If True, only report what would be deleted without actually deleting (default: True)    
    user_id : UUID, optional    
        Limit cleanup to specific user's data (default: None)  
    text_doc : bool  
        If True, use SQL-based filtering to find unused TextDocuments and call cognee.delete()  
        for proper whole-document deletion (default: False)  
        
    Returns    
    -------    
    Dict[str, Any]    
        Cleanup results with status, counts, and timestamp    
    """    
    logger.info(    
        "Starting cleanup task",    
        days_threshold=days_threshold,    
        dry_run=dry_run,    
        user_id=str(user_id) if user_id else None,  
        text_doc=text_doc  
    )    
        
    # Calculate cutoff timestamp  
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
      
    if text_doc:  
        # SQL-based approach: Find unused TextDocuments and use cognee.delete()  
        return await _cleanup_via_sql(cutoff_date, dry_run, user_id)  
    else:  
        # Graph-based approach: Find unused nodes directly from graph  
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
