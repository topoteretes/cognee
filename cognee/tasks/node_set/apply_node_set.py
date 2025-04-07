import uuid
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence, Protocol, Callable
from contextlib import asynccontextmanager

from cognee.shared.logging_utils import get_logger
from sqlalchemy.future import select
from sqlalchemy import or_

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
from cognee.modules.data.models import Data
from cognee.infrastructure.engine.models.DataPoint import DataPoint

# Configure logger
logger = get_logger(name="apply_node_set")


async def apply_node_set(data: Union[DataPoint, List[DataPoint]]) -> Union[DataPoint, List[DataPoint]]:
    """Apply NodeSet values to DataPoint objects.
    
    Args:
        data: Single DataPoint or list of DataPoints to process
        
    Returns:
        The processed DataPoint(s) with updated NodeSet values
    """
    if not data:
        logger.warning("No data provided to apply NodeSet values")
        return data

    # Convert single DataPoint to list for uniform processing
    data_points = data if isinstance(data, list) else [data]
    logger.info(f"Applying NodeSet values to {len(data_points)} DataPoints")
    
    # Process DataPoint objects to apply NodeSet values
    updated_data_points = await _process_data_points(data_points)
    
    # Create set nodes for each NodeSet
    await _create_set_nodes(updated_data_points)
    
    # Return data in the same format it was received
    return data_points if isinstance(data, list) else data_points[0]


async def _process_data_points(data_points: List[DataPoint]) -> List[DataPoint]:
    """Process DataPoint objects to apply NodeSet values from the database.
    
    Args:
        data_points: List of DataPoint objects to process
        
    Returns:
        The processed list of DataPoints with updated NodeSet values
    """
    try:
        if not data_points:
            return []

        # Extract IDs and collect document relationships
        data_point_ids, parent_doc_map = _collect_ids_and_relationships(data_points)
        
        # Get NodeSet values from database
        nodeset_map = await _fetch_nodesets_from_database(data_point_ids, parent_doc_map)
        
        # Apply NodeSet values to DataPoints
        _apply_nodesets_to_datapoints(data_points, nodeset_map, parent_doc_map)
        
        return data_points
    
    except Exception as e:
        logger.error(f"Error processing DataPoints: {str(e)}")
        return data_points


def _collect_ids_and_relationships(data_points: List[DataPoint]) -> Tuple[List[str], Dict[str, str]]:
    """Extract DataPoint IDs and document relationships.
    
    Args:
        data_points: List of DataPoint objects
        
    Returns:
        Tuple containing:
            - List of DataPoint IDs
            - Dictionary mapping DataPoint IDs to parent document IDs
    """
    data_point_ids = []
    parent_doc_ids = []
    parent_doc_map = {}
    
    # Collect all IDs to look up
    for dp in data_points:
        # Get the DataPoint ID
        if hasattr(dp, "id"):
            dp_id = str(dp.id)
            data_point_ids.append(dp_id)
        
            # Check if there's a parent document to get NodeSet from
            if (hasattr(dp, "made_from") and 
                hasattr(dp.made_from, "is_part_of") and 
                hasattr(dp.made_from.is_part_of, "id")):
                
                parent_id = str(dp.made_from.is_part_of.id)
                parent_doc_ids.append(parent_id)
                parent_doc_map[dp_id] = parent_id
    
    logger.info(f"Found {len(data_point_ids)} DataPoint IDs and {len(parent_doc_ids)} parent document IDs")
    
    # Combine all IDs for database lookup
    return data_point_ids + parent_doc_ids, parent_doc_map


async def _fetch_nodesets_from_database(ids: List[str], parent_doc_map: Dict[str, str]) -> Dict[str, Any]:
    """Fetch NodeSet values from the database for the given IDs.
    
    Args:
        ids: List of IDs to search for
        parent_doc_map: Dictionary mapping DataPoint IDs to parent document IDs
        
    Returns:
        Dictionary mapping document IDs to their NodeSet values
    """
    # Convert string IDs to UUIDs for database lookup
    uuid_objects = _convert_ids_to_uuids(ids)
    if not uuid_objects:
        return {}
    
    # Query the database for NodeSet values
    nodeset_map = {}
    
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as sess:
        # Query for records matching the IDs
        query = select(Data).where(Data.id.in_(uuid_objects))
        result = await sess.execute(query)
        records = result.scalars().all()
        
        logger.info(f"Found {len(records)} total records matching the IDs")
        
        # Extract NodeSet values from records
        for record in records:
            if record.node_set is not None:
                nodeset_map[str(record.id)] = record.node_set
        
        logger.info(f"Found {len(nodeset_map)} records with non-NULL NodeSet values")
    
    return nodeset_map


def _convert_ids_to_uuids(ids: List[str]) -> List[uuid.UUID]:
    """Convert string IDs to UUID objects.
    
    Args:
        ids: List of string IDs to convert
        
    Returns:
        List of UUID objects
    """
    uuid_objects = []
    for id_str in ids:
        try:
            uuid_objects.append(uuid.UUID(id_str))
        except Exception as e:
            logger.warning(f"Failed to convert ID {id_str} to UUID: {str(e)}")
    
    logger.info(f"Converted {len(uuid_objects)} out of {len(ids)} IDs to UUID objects")
    return uuid_objects


def _apply_nodesets_to_datapoints(
    data_points: List[DataPoint], 
    nodeset_map: Dict[str, Any], 
    parent_doc_map: Dict[str, str]
) -> None:
    """Apply NodeSet values to DataPoints.
    
    Args:
        data_points: List of DataPoint objects to update
        nodeset_map: Dictionary mapping document IDs to their NodeSet values
        parent_doc_map: Dictionary mapping DataPoint IDs to parent document IDs
    """
    for dp in data_points:
        dp_id = str(dp.id)
        
        # Try direct match first
        if dp_id in nodeset_map:
            nodeset = nodeset_map[dp_id]
            logger.info(f"Found NodeSet for {dp_id}: {nodeset}")
            dp.NodeSet = nodeset
            
        # Then try parent document
        elif dp_id in parent_doc_map and parent_doc_map[dp_id] in nodeset_map:
            parent_id = parent_doc_map[dp_id]
            nodeset = nodeset_map[parent_id]
            logger.info(f"Found NodeSet from parent document {parent_id} for {dp_id}: {nodeset}")
            dp.NodeSet = nodeset


async def _create_set_nodes(data_points: List[DataPoint]) -> None:
    """Create set nodes for DataPoints with NodeSets.
    
    Args:
        data_points: List of DataPoint objects to process
    """
    try:
        logger.info(f"Creating set nodes for {len(data_points)} DataPoints")
        
        for dp in data_points:
            if not hasattr(dp, "NodeSet") or not dp.NodeSet:
                continue
                
            try:
                # Create set nodes for the NodeSet (one per value)
                document_id = str(dp.id) if hasattr(dp, "id") else None
                set_node_ids, edge_ids = await create_set_node(dp.NodeSet, document_id=document_id)
                
                if set_node_ids and len(set_node_ids) > 0:
                    logger.info(f"Created {len(set_node_ids)} set nodes for NodeSet with {len(dp.NodeSet)} values")
                    
                    # Store the set node IDs with the DataPoint if possible
                    try:
                        # Store as JSON string if multiple IDs, or single ID if only one
                        if len(set_node_ids) > 1:
                            dp.SetNodeIds = json.dumps(set_node_ids)
                        else:
                            dp.SetNodeId = set_node_ids[0]
                    except Exception as e:
                        logger.warning(f"Failed to store set node IDs for NodeSet: {str(e)}")
                else:
                    logger.warning("Failed to create set nodes for NodeSet")
            except Exception as e:
                logger.error(f"Error creating set nodes: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing NodeSets: {str(e)}")


async def create_set_node(
    nodes: Union[List[str], str], 
    name: Optional[str] = None, 
    document_id: Optional[str] = None
) -> Tuple[Optional[List[str]], List[str]]:
    """Create individual nodes for each value in the NodeSet.
    
    Args:
        nodes: List of node values or JSON string representing node values
        name: Base name for the NodeSet (optional)
        document_id: ID of the document containing the NodeSet (optional)
        
    Returns:
        Tuple containing:
            - List of created set node IDs (or None if creation failed)
            - List of created edge IDs
    """
    try:
        if not nodes:
            logger.warning("No nodes provided to create set nodes")
            return None, []

        # Get the graph engine
        graph_engine = await get_graph_engine()

        # Parse nodes if provided as JSON string
        nodes_list = _parse_nodes_input(nodes)
        if not nodes_list:
            return None, []

        # Base name for the set if not provided
        base_name = name or f"NodeSet_{uuid.uuid4().hex[:8]}"
        logger.info(f"Creating individual set nodes for {len(nodes_list)} values with base name '{base_name}'")
        
        # Create set nodes using the appropriate strategy
        return await _create_set_nodes_unified(graph_engine, nodes_list, base_name, document_id)
            
    except Exception as e:
        logger.error(f"Failed to create set nodes: {str(e)}")
        return None, []


def _parse_nodes_input(nodes: Union[List[str], str]) -> Optional[List[str]]:
    """Parse the nodes input.
    
    Args:
        nodes: List of node values or JSON string representing node values
        
    Returns:
        List of node values or None if parsing failed
    """
    if isinstance(nodes, str):
        try:
            parsed_nodes = json.loads(nodes)
            logger.info(f"Parsed nodes string into list with {len(parsed_nodes)} items")
            return parsed_nodes
        except Exception as e:
            logger.error(f"Failed to parse nodes as JSON: {str(e)}")
            return None
    return nodes


async def _create_set_nodes_unified(
    graph_engine: Any,
    nodes_list: List[str],
    base_name: str,
    document_id: Optional[str]
) -> Tuple[List[str], List[str]]:
    """Create set nodes using either NetworkX or generic graph engine.
    
    Args:
        graph_engine: The graph engine instance
        nodes_list: List of node values
        base_name: Base name for the NodeSet
        document_id: ID of the document containing the NodeSet (optional)
        
    Returns:
        Tuple containing:
            - List of created set node IDs
            - List of created edge IDs
    """
    all_set_node_ids = []
    all_edge_ids = []
    
    # Define strategies for node and edge creation based on graph engine type
    if isinstance(graph_engine, NetworkXAdapter):
        # NetworkX-specific strategy
        async def create_node(node_value: str) -> str:
            set_node_id = str(uuid.uuid4())
            node_name = f"NodeSet_{node_value}_{uuid.uuid4().hex[:8]}"
            
            graph_engine.graph.add_node(
                set_node_id,
                id=set_node_id,
                type="NodeSet",
                name=node_name,
                node_id=node_value
            )
            
            # Validate node creation
            if set_node_id in graph_engine.graph.nodes():
                node_props = dict(graph_engine.graph.nodes[set_node_id])
                logger.info(f"Created set node for value '{node_value}': {json.dumps(node_props)}")
            else:
                logger.warning(f"Node {set_node_id} not found in graph after adding")
                
            return set_node_id
            
        async def create_value_edge(set_node_id: str, node_value: str) -> List[str]:
            edge_ids = []
            try:
                edge_id = str(uuid.uuid4())
                graph_engine.graph.add_edge(
                    set_node_id,
                    node_value,
                    id=edge_id,
                    type="CONTAINS"
                )
                edge_ids.append(edge_id)
            except Exception as e:
                logger.warning(f"Failed to create edge from set node to node {node_value}: {str(e)}")
            return edge_ids
            
        async def create_document_edge(document_id: str, set_node_id: str) -> List[str]:
            edge_ids = []
            try:
                doc_to_nodeset_id = str(uuid.uuid4())
                graph_engine.graph.add_edge(
                    document_id,
                    set_node_id,
                    id=doc_to_nodeset_id,
                    type="HAS_NODESET"
                )
                edge_ids.append(doc_to_nodeset_id)
                logger.info(f"Created edge from document {document_id} to NodeSet {set_node_id}")
            except Exception as e:
                logger.warning(f"Failed to create edge from document to NodeSet: {str(e)}")
            return edge_ids
            
        # Finalize function for NetworkX
        async def finalize() -> None:
            await graph_engine.save_graph_to_file(graph_engine.filename)
            
    else:
        # Generic graph engine strategy
        async def create_node(node_value: str) -> str:
            node_name = f"NodeSet_{node_value}_{uuid.uuid4().hex[:8]}"
            set_node_props = {
                "name": node_name,
                "type": "NodeSet",
                "node_id": node_value
            }
            return await graph_engine.create_node(set_node_props)
            
        async def create_value_edge(set_node_id: str, node_value: str) -> List[str]:
            edge_ids = []
            try:
                edge_id = await graph_engine.create_edge(
                    source_id=set_node_id,
                    target_id=node_value,
                    edge_type="CONTAINS"
                )
                edge_ids.append(edge_id)
            except Exception as e:
                logger.warning(f"Failed to create edge from set node to node {node_value}: {str(e)}")
            return edge_ids
            
        async def create_document_edge(document_id: str, set_node_id: str) -> List[str]:
            edge_ids = []
            try:
                doc_to_nodeset_id = await graph_engine.create_edge(
                    source_id=document_id,
                    target_id=set_node_id,
                    edge_type="HAS_NODESET"
                )
                edge_ids.append(doc_to_nodeset_id)
                logger.info(f"Created edge from document {document_id} to NodeSet {set_node_id}")
            except Exception as e:
                logger.warning(f"Failed to create edge from document to NodeSet: {str(e)}")
            return edge_ids
            
        # Finalize function for generic engine (no-op)
        async def finalize() -> None:
            pass
    
    # Unified process for both engine types
    for node_value in nodes_list:
        try:
            # Create the node
            set_node_id = await create_node(node_value)
            
            # Create edges to the value
            value_edge_ids = await create_value_edge(set_node_id, node_value)
            all_edge_ids.extend(value_edge_ids)
            
            # Create edges to the document if provided
            if document_id:
                doc_edge_ids = await create_document_edge(document_id, set_node_id)
                all_edge_ids.extend(doc_edge_ids)
                
            all_set_node_ids.append(set_node_id)
        except Exception as e:
            logger.error(f"Failed to create set node for value '{node_value}': {str(e)}")
    
    # Finalize the process
    await finalize()
    
    # Return results
    if all_set_node_ids:
        logger.info(f"Created {len(all_set_node_ids)} individual set nodes with values: {nodes_list}")
        return all_set_node_ids, all_edge_ids
    else:
        logger.error("Failed to create any set nodes")
        return [], []
