"""Create ontology method following Cognee patterns."""

from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    OntologyGraph,
    OntologyNode,
    OntologyEdge,
    OntologyScope,
    OntologyFormat,
    OntologyContext,
)
from cognee.modules.ontology.config import get_ontology_config
from cognee.modules.users.models import User

logger = get_logger("ontology.create")


async def create_ontology(
    ontology_data: Dict[str, Any],
    user: User,
    scope: OntologyScope = OntologyScope.USER,
    context: Optional[OntologyContext] = None
) -> OntologyGraph:
    """
    Create a new ontology from provided data.
    
    Args:
        ontology_data: Dictionary containing ontology structure
        user: User creating the ontology
        scope: Scope for the ontology (user, domain, global, etc.)
        context: Optional context for the ontology
    
    Returns:
        Created OntologyGraph instance
    
    Raises:
        ValueError: If ontology_data is invalid
        RuntimeError: If ontology creation fails
    """
    
    try:
        config = get_ontology_config()
        
        # Validate required fields
        if "nodes" not in ontology_data:
            raise ValueError("Ontology data must contain 'nodes' field")
        
        # Extract basic information
        ontology_id = ontology_data.get("id", str(uuid4()))
        ontology_name = ontology_data.get("name", f"ontology_{ontology_id}")
        description = ontology_data.get("description", "")
        format_type = OntologyFormat(ontology_data.get("format", config.default_format))
        
        # Parse nodes
        nodes = []
        for node_data in ontology_data["nodes"]:
            if not isinstance(node_data, dict):
                logger.warning(f"Skipping invalid node data: {node_data}")
                continue
                
            try:
                node = OntologyNode(
                    id=node_data.get("id", str(uuid4())),
                    name=node_data.get("name", "unnamed_node"),
                    type=node_data.get("type", "entity"),
                    description=node_data.get("description", ""),
                    category=node_data.get("category", "general"),
                    properties=node_data.get("properties", {})
                )
                nodes.append(node)
            except Exception as e:
                logger.warning(f"Failed to parse node {node_data.get('id', 'unknown')}: {e}")
                continue
        
        # Parse edges
        edges = []
        for edge_data in ontology_data.get("edges", []):
            if not isinstance(edge_data, dict):
                logger.warning(f"Skipping invalid edge data: {edge_data}")
                continue
                
            try:
                edge = OntologyEdge(
                    id=edge_data.get("id", str(uuid4())),
                    source_id=edge_data["source"],
                    target_id=edge_data["target"],
                    relationship_type=edge_data.get("relationship", "related_to"),
                    properties=edge_data.get("properties", {}),
                    weight=edge_data.get("weight")
                )
                edges.append(edge)
            except KeyError as e:
                logger.warning(f"Edge missing required field {e}: {edge_data}")
                continue
            except Exception as e:
                logger.warning(f"Failed to parse edge: {e}")
                continue
        
        # Create metadata
        metadata = ontology_data.get("metadata", {})
        metadata.update({
            "created_by": str(user.id),
            "created_at": datetime.now().isoformat(),
            "scope": scope.value,
            "format": format_type.value,
        })
        
        if context:
            metadata.update({
                "domain": context.domain,
                "pipeline_name": context.pipeline_name,
                "dataset_id": context.dataset_id,
            })
        
        # Create ontology
        ontology = OntologyGraph(
            id=ontology_id,
            name=ontology_name,
            description=description,
            format=format_type,
            scope=scope,
            nodes=nodes,
            edges=edges,
            metadata=metadata
        )
        
        logger.info(
            f"Created ontology '{ontology_name}' with {len(nodes)} nodes "
            f"and {len(edges)} edges for user {user.id}"
        )
        
        return ontology
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        error_msg = f"Failed to create ontology: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


async def create_empty_ontology(
    name: str,
    user: User,
    scope: OntologyScope = OntologyScope.USER,
    domain: Optional[str] = None,
    description: str = ""
) -> OntologyGraph:
    """
    Create an empty ontology with basic structure.
    
    Args:
        name: Name for the ontology
        user: User creating the ontology
        scope: Scope for the ontology
        domain: Optional domain for the ontology
        description: Optional description
    
    Returns:
        Empty OntologyGraph instance
    """
    
    config = get_ontology_config()
    ontology_id = str(uuid4())
    
    metadata = {
        "created_by": str(user.id),
        "created_at": datetime.now().isoformat(),
        "scope": scope.value,
        "format": config.default_format,
    }
    
    if domain:
        metadata["domain"] = domain
    
    ontology = OntologyGraph(
        id=ontology_id,
        name=name,
        description=description,
        format=OntologyFormat(config.default_format),
        scope=scope,
        nodes=[],
        edges=[],
        metadata=metadata
    )
    
    logger.info(f"Created empty ontology '{name}' for user {user.id}")
    return ontology
