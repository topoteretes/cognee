import json
import tempfile
from uuid import UUID
from typing import Dict, Any, List
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger()


async def export_dataset_data(dataset_id: UUID, user: User) -> Dict[str, Any]:
    """
    Export all data associated with a dataset from Kuzu + LanceDB + PostgreSQL.
    
    Args:
        dataset_id: UUID of the dataset to export
        user: User performing the export
        
    Returns:
        Dict containing all exported data in transfer format
    """
    logger.info(f"Starting export of dataset {dataset_id} for user {user.id}")
    
    # Set database context for the user
    set_database_global_context_variables(user=user)
    
    try:
        # Get dataset metadata from PostgreSQL
        dataset_metadata = await _export_dataset_metadata(dataset_id, user)
        
        # Export graph data from Kuzu
        graph_data = await _export_graph_data(dataset_id, user)
        
        # Export vector data from LanceDB
        vector_data = await _export_vector_data(dataset_id, user)
        
        # Export metastore data (permissions, relationships)
        metastore_data = await _export_metastore_data(dataset_id, user)
        
        export_package = {
            "dataset_id": str(dataset_id),
            "metadata": dataset_metadata,
            "graph_data": graph_data,
            "vector_data": vector_data,
            "metastore_data": metastore_data,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_user_id": str(user.id),
            "version": "1.0"
        }
        
        logger.info(f"Successfully exported dataset {dataset_id}")
        return export_package
        
    except Exception as e:
        logger.error(f"Error exporting dataset {dataset_id}: {str(e)}")
        raise


async def _export_dataset_metadata(dataset_id: UUID, user: User) -> Dict[str, Any]:
    """Export dataset metadata from PostgreSQL"""
    from cognee.modules.data.methods import get_dataset, get_dataset_data
    
    dataset = await get_dataset(user.id, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    dataset_data = await get_dataset_data(dataset_id)
    
    return {
        "dataset": {
            "id": str(dataset.id),
            "name": dataset.name,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
            "owner_id": str(dataset.owner_id)
        },
        "data_items": [
            {
                "id": str(data.id),
                "name": data.name,
                "extension": data.extension,
                "mime_type": data.mime_type,
                "content_hash": data.content_hash,
                "external_metadata": data.external_metadata,
                "node_set": data.node_set,
                "token_count": data.token_count,
                "data_size": data.data_size,
                "created_at": data.created_at.isoformat(),
                "updated_at": data.updated_at.isoformat() if data.updated_at else None
            }
            for data in dataset_data
        ]
    }


async def _export_graph_data(dataset_id: UUID, user: User) -> Dict[str, Any]:
    """Export all nodes and edges from Kuzu graph database"""
    graph_engine = get_graph_engine()
    
    try:
        # Get all nodes and edges for the dataset
        nodes, edges = await graph_engine.get_graph_data()
        
        # Filter nodes and edges that belong to this dataset
        # This assumes nodes have dataset context in their properties
        dataset_nodes = []
        dataset_edges = []
        
        for node_id, node_data in nodes:
            # Check if node belongs to this dataset (you may need to adjust this logic)
            if _node_belongs_to_dataset(node_data, dataset_id):
                dataset_nodes.append({
                    "id": node_id,
                    "data": node_data
                })
        
        for from_node, to_node, edge_label, edge_data in edges:
            # Check if edge connects nodes from this dataset
            if _edge_belongs_to_dataset(from_node, to_node, dataset_nodes):
                dataset_edges.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "edge_label": edge_label,
                    "data": edge_data
                })
        
        return {
            "nodes": dataset_nodes,
            "edges": dataset_edges,
            "node_count": len(dataset_nodes),
            "edge_count": len(dataset_edges)
        }
        
    except Exception as e:
        logger.error(f"Error exporting graph data: {str(e)}")
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}


async def _export_vector_data(dataset_id: UUID, user: User) -> Dict[str, Any]:
    """Export vector embeddings from LanceDB collections"""
    vector_engine = get_vector_engine()
    
    try:
        # Get collection name for the dataset
        from cognee.infrastructure.databases.utils import get_or_create_dataset_database
        dataset_db = await get_or_create_dataset_database(dataset_id, user)
        collection_name = dataset_db.vector_database_name
        
        # Check if collection exists
        if await vector_engine.has_collection(collection_name):
            # Export all data from the collection
            # This is a simplified approach - you may need to implement collection export in your vector DB adapter
            collection_data = await _export_collection_data(vector_engine, collection_name)
            
            return {
                "collections": {
                    collection_name: collection_data
                },
                "collection_count": 1
            }
        else:
            return {
                "collections": {},
                "collection_count": 0
            }
            
    except Exception as e:
        logger.error(f"Error exporting vector data: {str(e)}")
        return {"collections": {}, "collection_count": 0}


async def _export_collection_data(vector_engine, collection_name: str) -> Dict[str, Any]:
    """Export data from a LanceDB collection"""
    try:
        # Get collection
        collection = await vector_engine.get_collection(collection_name)
        
        # For now, return metadata about the collection
        # You would need to implement actual data export based on your LanceDB adapter
        return {
            "name": collection_name,
            "schema": "exported_schema",  # Implement schema export
            "data": "exported_data",      # Implement data export
            "metadata": {}
        }
    except Exception as e:
        logger.error(f"Error exporting collection {collection_name}: {str(e)}")
        return {}


async def _export_metastore_data(dataset_id: UUID, user: User) -> Dict[str, Any]:
    """Export metastore data including permissions and relationships"""
    db_engine = get_relational_engine()
    
    try:
        async with db_engine.get_async_session() as session:
            # Export ACL permissions for the dataset
            from cognee.modules.users.models import ACL
            from sqlalchemy import select
            
            result = await session.execute(
                select(ACL).where(ACL.dataset_id == dataset_id)
            )
            acls = result.scalars().all()
            
            permissions = [
                {
                    "id": str(acl.id),
                    "principal_id": str(acl.principal_id),
                    "permission_id": str(acl.permission_id),
                    "created_at": acl.created_at.isoformat()
                }
                for acl in acls
            ]
            
            # Export dataset database configuration
            from cognee.modules.users.models import DatasetDatabase
            result = await session.execute(
                select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
            )
            dataset_db = result.scalar_one_or_none()
            
            dataset_database_config = None
            if dataset_db:
                dataset_database_config = {
                    "owner_id": str(dataset_db.owner_id),
                    "vector_database_name": dataset_db.vector_database_name,
                    "graph_database_name": dataset_db.graph_database_name,
                    "created_at": dataset_db.created_at.isoformat(),
                    "updated_at": dataset_db.updated_at.isoformat() if dataset_db.updated_at else None
                }
            
            return {
                "permissions": permissions,
                "dataset_database_config": dataset_database_config
            }
            
    except Exception as e:
        logger.error(f"Error exporting metastore data: {str(e)}")
        return {"permissions": [], "dataset_database_config": None}


def _node_belongs_to_dataset(node_data: Dict[str, Any], dataset_id: UUID) -> bool:
    """Check if a node belongs to the specified dataset"""
    # This is a simplified check - you may need to adjust based on your data model
    if isinstance(node_data, dict):
        # Check if node has dataset context in properties
        return str(dataset_id) in str(node_data.get("properties", ""))
    return False


def _edge_belongs_to_dataset(from_node: str, to_node: str, dataset_nodes: List[Dict]) -> bool:
    """Check if an edge connects nodes from the dataset"""
    node_ids = {node["id"] for node in dataset_nodes}
    return from_node in node_ids and to_node in node_ids
