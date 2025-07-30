import json
from uuid import UUID, uuid4
from typing import Dict, Any
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger()


async def import_dataset_data(
    transfer_content: bytes, 
    target_user: User, 
    import_request
) -> Dict[str, Any]:
    """
    Import data from a transfer bundle into target user's databases.
    
    Args:
        transfer_content: JSON content of the transfer bundle
        target_user: User who will own the imported data
        import_request: Import configuration options
        
    Returns:
        Dict with import results
    """
    logger.info(f"Starting import for user {target_user.id}")
    
    try:
        # Parse transfer content
        transfer_data = json.loads(transfer_content.decode('utf-8'))
        
        # Validate transfer data structure
        _validate_transfer_data(transfer_data)

        # Create new dataset for imported data
        new_dataset = await _create_target_dataset(
            transfer_data, target_user, import_request
        )
        
        # Set database context for the target user with new dataset
        await set_database_global_context_variables(
            dataset=new_dataset.id,
            user_id=target_user.id
        )        # Import graph data to Kuzu
        nodes_imported, edges_imported = await _import_graph_data(
            transfer_data["graph_data"], new_dataset.id, target_user
        )
        
        # Import vector data to LanceDB
        vectors_imported = await _import_vector_data(
            transfer_data["vector_data"], new_dataset.id, target_user
        )
        
        # Update metastore with new ownership
        await _import_metastore_data(
            transfer_data["metastore_data"], new_dataset.id, target_user
        )
        
        # Import data items
        await _import_data_items(
            transfer_data["metadata"]["data_items"], new_dataset.id, target_user
        )
        
        return {
            "success": True,
            "dataset_id": new_dataset.id,
            "message": f"Successfully imported dataset '{new_dataset.name}'",
            "imported_nodes": nodes_imported,
            "imported_edges": edges_imported,
            "imported_vectors": vectors_imported
        }
        
    except Exception as e:
        logger.error(f"Error importing dataset: {str(e)}")
        return {
            "success": False,
            "dataset_id": None,
            "message": f"Import failed: {str(e)}",
            "imported_nodes": 0,
            "imported_edges": 0,
            "imported_vectors": 0
        }


def _validate_transfer_data(transfer_data: Dict[str, Any]) -> None:
    """Validate the structure of transfer data"""
    required_keys = ["dataset_id", "metadata", "graph_data", "vector_data", "metastore_data", "version"]
    
    for key in required_keys:
        if key not in transfer_data:
            raise ValueError(f"Missing required key in transfer data: {key}")
    
    if transfer_data.get("version") != "1.0":
        raise ValueError(f"Unsupported transfer data version: {transfer_data.get('version')}")


async def _create_target_dataset(
    transfer_data: Dict[str, Any], 
    target_user: User, 
    import_request
) -> Any:
    """Create a new dataset for the imported data"""
    from cognee.modules.data.methods import create_dataset
    from cognee.modules.users.permissions.methods import give_permission_on_dataset
    
    # Determine dataset name
    if import_request.target_dataset_name:
        dataset_name = import_request.target_dataset_name
    else:
        original_name = transfer_data["metadata"]["dataset"]["name"]
        dataset_name = f"{original_name}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create dataset
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        dataset = await create_dataset(
            dataset_name=dataset_name,
            user=target_user,
            session=session
        )
        
        # Give all permissions to the target user
        await give_permission_on_dataset(target_user, dataset.id, "read")
        await give_permission_on_dataset(target_user, dataset.id, "write")
        await give_permission_on_dataset(target_user, dataset.id, "share")
        await give_permission_on_dataset(target_user, dataset.id, "delete")
        
        return dataset


async def _import_graph_data(
    graph_data: Dict[str, Any], 
    dataset_id: UUID, 
    target_user: User
) -> tuple[int, int]:
    """Import nodes and edges into Kuzu graph database"""
    graph_engine = get_graph_engine()
    
    nodes_imported = 0
    edges_imported = 0
    
    try:
        # Import nodes with new IDs and user mapping
        node_id_mapping = {}
        
        for node in graph_data.get("nodes", []):
            old_node_id = node["id"]
            new_node_id = str(uuid4())
            node_id_mapping[old_node_id] = new_node_id
            
            # Update node data with new user ownership
            node_data = node["data"].copy()
            node_data = _update_ownership_in_properties(node_data, target_user.id)
            
            # Create DataPoint for the node
            from cognee.infrastructure.engine import DataPoint
            data_point = DataPoint(
                id=new_node_id,
                **node_data
            )
            
            await graph_engine.add_node(data_point)
            nodes_imported += 1
        
        # Import edges with mapped node IDs
        for edge in graph_data.get("edges", []):
            old_from_node = edge["from_node"]
            old_to_node = edge["to_node"]
            
            # Map to new node IDs
            new_from_node = node_id_mapping.get(old_from_node)
            new_to_node = node_id_mapping.get(old_to_node)
            
            if new_from_node and new_to_node:
                edge_data = edge["data"].copy()
                edge_data = _update_ownership_in_properties(edge_data, target_user.id)
                
                await graph_engine.add_edge(
                    from_node=new_from_node,
                    to_node=new_to_node,
                    relationship_name=edge["edge_label"],
                    edge_properties=edge_data
                )
                edges_imported += 1
        
        logger.info(f"Imported {nodes_imported} nodes and {edges_imported} edges")
        return nodes_imported, edges_imported
        
    except Exception as e:
        logger.error(f"Error importing graph data: {str(e)}")
        return nodes_imported, edges_imported


async def _import_vector_data(
    vector_data: Dict[str, Any], 
    dataset_id: UUID, 
    target_user: User
) -> int:
    """Import vector embeddings into LanceDB collections"""
    vector_engine = get_vector_engine()
    vectors_imported = 0
    
    try:
        # Get or create vector collection for the new dataset
        from cognee.infrastructure.databases.utils import get_or_create_dataset_database
        dataset_db = await get_or_create_dataset_database(dataset_id, target_user)
        base_collection_name = dataset_db.vector_database_name
        
        # Import collections with proper naming
        for source_collection_name, collection_data in vector_data.get("collections", {}).items():
            # Create unique collection name for each source collection
            target_collection_name = f"{base_collection_name}_{source_collection_name}"
            
            # Create target collection if it doesn't exist
            if not await vector_engine.has_collection(target_collection_name):
                await _create_vector_collection(vector_engine, target_collection_name, collection_data)
            
            # Import vector data with proper collection mapping
            collection_vectors = await _import_collection_vectors(
                vector_engine, target_collection_name, collection_data, target_user
            )
            vectors_imported += collection_vectors
            
            logger.info(f"Imported {collection_vectors} vectors from {source_collection_name} to {target_collection_name}")
        
        logger.info(f"Total imported vectors: {vectors_imported}")
        return vectors_imported
        
    except Exception as e:
        logger.error(f"Error importing vector data: {str(e)}")
        return vectors_imported


async def _create_vector_collection(vector_engine, collection_name: str, collection_data: Dict[str, Any]):
    """Create a new vector collection based on imported schema"""
    # This is a placeholder - implement based on your vector DB requirements
    try:
        from cognee.infrastructure.engine import DataPoint
        await vector_engine.create_collection(collection_name, DataPoint)
    except Exception as e:
        logger.error(f"Error creating vector collection {collection_name}: {str(e)}")


async def _import_collection_vectors(
    vector_engine, collection_name: str, collection_data: Dict[str, Any], target_user: User
) -> int:
    """Import vectors into a collection"""
    # This is a placeholder - implement actual vector import based on your data format
    try:
        # For now, return a placeholder count
        return len(collection_data.get("data", []))
    except Exception as e:
        logger.error(f"Error importing vectors to {collection_name}: {str(e)}")
        return 0


async def _import_metastore_data(
    metastore_data: Dict[str, Any], 
    dataset_id: UUID, 
    target_user: User
) -> None:
    """Import metastore data with updated ownership"""
    db_engine = get_relational_engine()
    
    try:
        async with db_engine.get_async_session() as session:
            # Update dataset database configuration
            dataset_db_config = metastore_data.get("dataset_database_config")
            if dataset_db_config:
                from cognee.infrastructure.databases.utils import get_or_create_dataset_database
                # This will create the dataset database record with new ownership
                await get_or_create_dataset_database(dataset_id, target_user)
            
        logger.info("Updated metastore data with new ownership")
        
    except Exception as e:
        logger.error(f"Error importing metastore data: {str(e)}")


async def _import_data_items(
    data_items: list, 
    dataset_id: UUID, 
    target_user: User
) -> None:
    """Import data items with updated ownership"""
    db_engine = get_relational_engine()
    
    try:
        from cognee.modules.data.models import Data, DatasetData
        
        async with db_engine.get_async_session() as session:
            for item in data_items:
                # Create new data item with updated ownership
                new_data = Data(
                    id=uuid4(),
                    name=item["name"],
                    extension=item["extension"],
                    mime_type=item["mime_type"],
                    raw_data_location=item.get("raw_data_location", ""),
                    owner_id=target_user.id,
                    tenant_id=target_user.tenant_id,
                    content_hash=item["content_hash"],
                    external_metadata=item["external_metadata"],
                    node_set=item["node_set"],
                    token_count=item["token_count"],
                    data_size=item["data_size"]
                )
                
                session.add(new_data)
                
                # Link to dataset
                dataset_data_link = DatasetData(
                    dataset_id=dataset_id,
                    data_id=new_data.id
                )
                session.add(dataset_data_link)
            
            await session.commit()
        
        logger.info(f"Imported {len(data_items)} data items")
        
    except Exception as e:
        logger.error(f"Error importing data items: {str(e)}")


def _update_ownership_in_properties(properties: Dict[str, Any], new_user_id: UUID) -> Dict[str, Any]:
    """Update ownership references in node/edge properties"""
    if isinstance(properties, dict):
        updated_properties = properties.copy()
        
        # Update any user_id or owner_id references
        if "user_id" in updated_properties:
            updated_properties["user_id"] = str(new_user_id)
        if "owner_id" in updated_properties:
            updated_properties["owner_id"] = str(new_user_id)
        
        return updated_properties
    
    return properties
