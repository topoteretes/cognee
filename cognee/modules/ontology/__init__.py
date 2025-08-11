"""
Ontology module for Cognee.

Provides ontology management capabilities including loading, processing,
and integration with pipelines following Cognee's architectural patterns.
"""

# Configuration (following Cognee pattern)
from .config import get_ontology_config

# Core data models
from .interfaces import (
    OntologyNode,
    OntologyEdge,
    OntologyGraph,
    OntologyContext,
    DataPointMapping,
    GraphBindingConfig,
    OntologyFormat,
    OntologyScope,
)

# Core implementations
from .manager import OntologyManager, create_ontology_manager
from .registry import OntologyRegistry
from .providers import (
    RDFOntologyProvider,
    JSONOntologyProvider,
    CSVOntologyProvider,
)

# Methods (following Cognee pattern)
from .methods import (
    create_ontology,
    get_ontology,
    load_ontology,
    register_ontology,
    delete_ontology,
)

# Legacy compatibility
from .rdf_xml.OntologyResolver import OntologyResolver


# Convenience functions for quick setup
async def create_ontology_system(
    config_file: str = None,
    use_database_registry: bool = False,
    enable_semantic_search: bool = False
) -> OntologyManager:
    """
    Create a fully configured ontology system.
    
    Args:
        config_file: Optional configuration file to load
        use_database_registry: Whether to use database-backed registry
        enable_semantic_search: Whether to enable semantic search capabilities
    
    Returns:
        Configured OntologyManager instance
    """
    # Create registry
    if use_database_registry:
        registry = DatabaseOntologyRegistry()
    else:
        registry = OntologyRegistry()
    
    # Create providers
    providers = {
        "json_provider": JSONOntologyProvider(),
        "csv_provider": CSVOntologyProvider(),
    }
    
    # Add RDF provider if available
    rdf_provider = RDFOntologyProvider()
    if rdf_provider.available:
        providers["rdf_provider"] = rdf_provider
    
    # Create adapters
    adapters = {
        "default_adapter": DefaultOntologyAdapter(),
        "graph_adapter": GraphOntologyAdapter(),
    }
    
    # Add semantic adapter if requested and available
    if enable_semantic_search:
        semantic_adapter = SemanticOntologyAdapter()
        if semantic_adapter.embeddings_available:
            adapters["semantic_adapter"] = semantic_adapter
    
    # Create resolver and binder
    datapoint_resolver = DefaultDataPointResolver()
    graph_binder = DefaultGraphBinder()
    
    # Create manager
    manager = await create_ontology_manager(
        registry=registry,
        providers=providers,
        adapters=adapters,
        datapoint_resolver=datapoint_resolver,
        graph_binder=graph_binder,
    )
    
    # Load configuration if provided
    if config_file:
        config = get_ontology_config()
        config.load_from_file(config_file)
        
        # Apply configurations to manager
        for domain, domain_config in config.domain_configs.items():
            manager.configure_datapoint_mapping(
                domain, domain_config["datapoint_mappings"]
            )
            manager.configure_graph_binding(
                domain, domain_config["graph_binding"]
            )
    
    return manager


def create_pipeline_injector(
    ontology_manager: OntologyManager,
    pipeline_name: str,
    domain: str = None
) -> OntologyInjector:
    """
    Create an ontology injector for a specific pipeline.
    
    Args:
        ontology_manager: The ontology manager instance
        pipeline_name: Name of the pipeline
        domain: Domain for the pipeline (optional)
    
    Returns:
        Configured OntologyInjector
    """
    configurator = PipelineOntologyConfigurator(ontology_manager)
    
    # Use pre-configured domain setup if available
    if domain in ["medical", "legal", "code"]:
        if domain == "medical":
            config = create_medical_pipeline_config()
        elif domain == "legal":
            config = create_legal_pipeline_config()
        elif domain == "code":
            config = create_code_pipeline_config()
        
        configurator.configure_pipeline(
            pipeline_name=pipeline_name,
            domain=config["domain"],
            datapoint_mappings=config["datapoint_mappings"],
            graph_binding=config["graph_binding"],
            task_specific_configs=config["task_configs"]
        )
    
    return configurator.create_ontology_injector(pipeline_name)


# Export following Cognee pattern
__all__ = [
    # Configuration
    "get_ontology_config",
    
    # Core classes
    "OntologyManager",
    "OntologyRegistry",
    
    # Data models
    "OntologyNode",
    "OntologyEdge", 
    "OntologyGraph",
    "OntologyContext",
    "DataPointMapping",
    "GraphBindingConfig",
    "OntologyFormat",
    "OntologyScope",
    
    # Providers
    "RDFOntologyProvider",
    "JSONOntologyProvider",
    "CSVOntologyProvider",
    
    # Methods
    "create_ontology",
    "get_ontology",
    "load_ontology",
    "register_ontology",
    "delete_ontology",
    
    # Convenience functions
    "create_ontology_system",
    "create_pipeline_injector",
    
    # Legacy compatibility
    "OntologyResolver",
]
