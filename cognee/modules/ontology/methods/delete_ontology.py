"""Delete ontology method following Cognee patterns."""

from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import OntologyContext
from cognee.modules.ontology.registry import OntologyRegistry
from cognee.modules.users.models import User

logger = get_logger("ontology.delete")


async def delete_ontology(
    ontology_id: str,
    user: User,
    context: Optional[OntologyContext] = None
) -> bool:
    """
    Delete ontology following Cognee delete patterns.
    
    Args:
        ontology_id: ID of the ontology to delete
        user: User requesting deletion
        context: Optional context for access control
    
    Returns:
        True if deletion successful, False otherwise
    """
    
    try:
        # This would use dependency injection in real implementation
        registry = OntologyRegistry()
        
        # Get ontology first to check existence and permissions
        ontology = await registry.get_ontology(ontology_id, context)
        
        if ontology is None:
            logger.warning(f"Ontology {ontology_id} not found for deletion")
            return False
        
        # TODO: Add access control check
        # For now, assume user has access
        
        # Perform deletion
        success = await registry.unregister_ontology(ontology_id, context)
        
        if success:
            logger.info(f"Deleted ontology {ontology_id} for user {user.id}")
        else:
            logger.error(f"Failed to delete ontology {ontology_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error deleting ontology {ontology_id}: {e}")
        return False
