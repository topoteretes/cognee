"""Load ontology method following Cognee patterns."""

from typing import Union, Dict, Any, Optional

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import OntologyGraph, OntologyContext
from cognee.modules.ontology.providers import JSONOntologyProvider, RDFOntologyProvider, CSVOntologyProvider
from cognee.modules.ontology.config import get_ontology_config
from cognee.modules.users.models import User

logger = get_logger("ontology.load")


async def load_ontology(
    source: Union[str, Dict[str, Any]],
    user: User,
    context: Optional[OntologyContext] = None
) -> Optional[OntologyGraph]:
    """
    Load ontology from various sources following Cognee patterns.
    
    Args:
        source: File path, URL, or data dictionary
        user: User loading the ontology
        context: Optional context for the ontology
    
    Returns:
        Loaded OntologyGraph or None if loading failed
    """
    
    try:
        config = get_ontology_config()
        
        # Determine provider based on source
        provider = None
        
        if isinstance(source, str):
            # File path or URL
            if source.endswith(('.owl', '.rdf', '.xml')) and config.rdf_provider_enabled:
                provider = RDFOntologyProvider()
                if not provider.available:
                    logger.warning("RDF provider not available, falling back to JSON")
                    provider = None
            elif source.endswith('.json') and config.json_provider_enabled:
                provider = JSONOntologyProvider()
            elif source.endswith('.csv') and config.csv_provider_enabled:
                provider = CSVOntologyProvider()
            else:
                # Default to JSON provider
                provider = JSONOntologyProvider()
        else:
            # Dictionary data - use JSON provider
            provider = JSONOntologyProvider()
        
        if provider is None:
            logger.error(f"No suitable provider found for source: {source}")
            return None
        
        # Load ontology
        ontology = await provider.load_ontology(source, context)
        
        # Validate ontology
        if not await provider.validate_ontology(ontology):
            logger.error(f"Ontology validation failed for source: {source}")
            return None
        
        # Add metadata about loading
        ontology.metadata.update({
            "loaded_by": str(user.id),
            "source": str(source),
            "provider": provider.__class__.__name__,
        })
        
        logger.info(
            f"Loaded ontology '{ontology.name}' with {len(ontology.nodes)} nodes "
            f"from source: {source}"
        )
        
        return ontology
        
    except Exception as e:
        logger.error(f"Failed to load ontology from {source}: {e}")
        return None
