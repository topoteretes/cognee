"""Get ontology method following Cognee patterns."""

from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import OntologyGraph, OntologyContext
from cognee.modules.ontology.registry import OntologyRegistry
from cognee.modules.users.models import User

logger = get_logger("ontology.get")


async def get_ontology(
    ontology_id: str,
    user: User,
    context: Optional[OntologyContext] = None
) -> Optional[OntologyGraph]:
    """
    Get ontology by ID following Cognee get patterns.
    
    Args:
        ontology_id: ID of the ontology to retrieve
        user: User requesting the ontology
        context: Optional context for access control
    
    Returns:
        OntologyGraph if found and accessible, None otherwise
    """
    
    try:
        # This would use dependency injection in real implementation
        registry = OntologyRegistry()
        
        ontology = await registry.get_ontology(ontology_id, context)
        
        if ontology is None:
            logger.info(f"Ontology {ontology_id} not found")
            return None
        
        # TODO: Add access control check based on user and ontology ownership
        # For now, assume user has access
        
        logger.info(f"Retrieved ontology {ontology_id} for user {user.id}")
        return ontology
        
    except Exception as e:
        logger.error(f"Failed to get ontology {ontology_id}: {e}")
        return None
