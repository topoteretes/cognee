"""Register ontology method following Cognee patterns."""

from typing import Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import OntologyGraph, OntologyScope, OntologyContext
from cognee.modules.ontology.registry import OntologyRegistry
from cognee.modules.users.models import User

logger = get_logger("ontology.register")


async def register_ontology(
    ontology: OntologyGraph,
    user: User,
    scope: OntologyScope = OntologyScope.USER,
    context: Optional[OntologyContext] = None
) -> str:
    """
    Register ontology in the registry following Cognee patterns.
    
    Args:
        ontology: OntologyGraph to register
        user: User registering the ontology
        scope: Scope for the ontology
        context: Optional context for registration
    
    Returns:
        Ontology ID if registration successful
        
    Raises:
        ValueError: If ontology is invalid
        RuntimeError: If registration fails
    """
    
    try:
        # Validate ontology
        if not ontology.nodes:
            raise ValueError("Cannot register empty ontology")
        
        # Update metadata
        ontology.metadata.update({
            "registered_by": str(user.id),
            "scope": scope.value,
        })
        
        if context:
            ontology.metadata.update({
                "domain": context.domain,
                "pipeline_name": context.pipeline_name,
                "dataset_id": context.dataset_id,
            })
        
        # This would use dependency injection in real implementation
        registry = OntologyRegistry()
        
        # Register in registry
        ontology_id = await registry.register_ontology(ontology, scope, context)
        
        logger.info(
            f"Registered ontology '{ontology.name}' (ID: {ontology_id}) "
            f"with scope {scope.value} for user {user.id}"
        )
        
        return ontology_id
        
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        error_msg = f"Failed to register ontology '{ontology.name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
