"""Core ontology manager implementation."""

import importlib
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
from uuid import uuid4

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IOntologyManager,
    IOntologyRegistry,
    IOntologyProvider,
    IOntologyAdapter,
    IDataPointResolver,
    IGraphBinder,
    OntologyGraph,
    OntologyNode,
    OntologyContext,
    DataPointMapping,
    GraphBindingConfig,
    OntologyScope,
)

logger = get_logger("OntologyManager")


class OntologyManager(IOntologyManager):
    """Core implementation of ontology management."""

    def __init__(
        self,
        registry: IOntologyRegistry,
        providers: Dict[str, IOntologyProvider],
        adapters: Dict[str, IOntologyAdapter],
        datapoint_resolver: IDataPointResolver,
        graph_binder: IGraphBinder,
    ):
        self.registry = registry
        self.providers = providers
        self.adapters = adapters
        self.datapoint_resolver = datapoint_resolver
        self.graph_binder = graph_binder
        
        # Domain-specific configurations
        self.domain_datapoint_mappings: Dict[str, List[DataPointMapping]] = {}
        self.domain_graph_bindings: Dict[str, GraphBindingConfig] = {}
        
        # Custom resolvers and binding strategies
        self.custom_resolvers: Dict[str, Callable] = {}
        self.custom_binding_strategies: Dict[str, Callable] = {}

    async def get_applicable_ontologies(
        self,
        context: OntologyContext
    ) -> List[OntologyGraph]:
        """Get ontologies applicable to given context."""
        ontologies = []
        
        # Get global ontologies
        global_ontologies = await self.registry.find_ontologies(
            scope=OntologyScope.GLOBAL,
            context=context
        )
        ontologies.extend(global_ontologies)
        
        # Get domain-specific ontologies
        if context.domain:
            domain_ontologies = await self.registry.find_ontologies(
                scope=OntologyScope.DOMAIN,
                domain=context.domain,
                context=context
            )
            ontologies.extend(domain_ontologies)
        
        # Get pipeline-specific ontologies
        if context.pipeline_name:
            pipeline_ontologies = await self.registry.find_ontologies(
                scope=OntologyScope.PIPELINE,
                context=context
            )
            ontologies.extend(pipeline_ontologies)
        
        # Get user-specific ontologies
        if context.user_id:
            user_ontologies = await self.registry.find_ontologies(
                scope=OntologyScope.USER,
                context=context
            )
            ontologies.extend(user_ontologies)
        
        # Get dataset-specific ontologies
        if context.dataset_id:
            dataset_ontologies = await self.registry.find_ontologies(
                scope=OntologyScope.DATASET,
                context=context
            )
            ontologies.extend(dataset_ontologies)
        
        # Remove duplicates and prioritize by scope
        unique_ontologies = self._prioritize_ontologies(ontologies)
        
        logger.info(f"Found {len(unique_ontologies)} applicable ontologies for context")
        return unique_ontologies

    async def enhance_with_ontology(
        self,
        content: str,
        context: OntologyContext
    ) -> Dict[str, Any]:
        """Enhance content with ontological information."""
        applicable_ontologies = await self.get_applicable_ontologies(context)
        
        enhanced_data = {
            "original_content": content,
            "ontological_annotations": [],
            "extracted_entities": [],
            "semantic_relationships": [],
        }
        
        for ontology in applicable_ontologies:
            adapter_name = self._get_adapter_for_ontology(ontology)
            if adapter_name not in self.adapters:
                logger.warning(f"No adapter found for ontology {ontology.id}")
                continue
            
            adapter = self.adapters[adapter_name]
            
            # Find matching nodes in content
            matching_nodes = await adapter.find_matching_nodes(
                content, ontology, similarity_threshold=0.7
            )
            
            for node in matching_nodes:
                enhanced_data["extracted_entities"].append({
                    "node_id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "category": node.category,
                    "ontology_id": ontology.id,
                    "confidence": 0.8,  # This should come from the adapter
                })
                
                # Get relationships for this node
                relationships = await adapter.get_node_relationships(
                    node.id, ontology, max_depth=1
                )
                
                for rel in relationships:
                    enhanced_data["semantic_relationships"].append({
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "relationship": rel.relationship_type,
                        "ontology_id": ontology.id,
                    })
        
        return enhanced_data

    async def inject_ontology_into_task(
        self,
        task_name: str,
        task_params: Dict[str, Any],
        context: OntologyContext
    ) -> Dict[str, Any]:
        """Inject ontological context into task parameters."""
        applicable_ontologies = await self.get_applicable_ontologies(context)
        
        # Merge all applicable ontologies
        if len(applicable_ontologies) > 1:
            primary_adapter = list(self.adapters.values())[0]  # Use first available adapter
            merged_ontology = await primary_adapter.merge_ontologies(applicable_ontologies)
            ontologies_to_inject = [merged_ontology]
        else:
            ontologies_to_inject = applicable_ontologies
        
        # Inject ontology-specific parameters
        enhanced_params = task_params.copy()
        enhanced_params["ontology_context"] = {
            "ontologies": [ont.id for ont in ontologies_to_inject],
            "domain": context.domain,
            "pipeline_name": context.pipeline_name,
        }
        
        # Add ontology-aware configurations
        if context.domain and context.domain in self.domain_datapoint_mappings:
            enhanced_params["datapoint_mappings"] = self.domain_datapoint_mappings[context.domain]
        
        if context.domain and context.domain in self.domain_graph_bindings:
            enhanced_params["graph_binding_config"] = self.domain_graph_bindings[context.domain]
        
        return enhanced_params

    async def resolve_to_datapoints(
        self,
        ontology_nodes: List[OntologyNode],
        context: OntologyContext
    ) -> List[Any]:  # List[DataPoint]
        """Resolve ontology nodes to DataPoint instances."""
        datapoints = []
        
        # Get domain-specific mappings
        mappings = self.domain_datapoint_mappings.get(context.domain, [])
        if not mappings:
            logger.warning(f"No DataPoint mappings configured for domain: {context.domain}")
            return datapoints
        
        for node in ontology_nodes:
            # Find appropriate mapping for this node type
            mapping = self._find_mapping_for_node(node, mappings)
            if not mapping:
                logger.debug(f"No mapping found for node type: {node.type}")
                continue
            
            try:
                datapoint = await self.datapoint_resolver.resolve_to_datapoint(
                    node, mapping, context
                )
                if datapoint:
                    datapoints.append(datapoint)
            except Exception as e:
                logger.error(f"Failed to resolve node {node.id} to DataPoint: {e}")
        
        logger.info(f"Resolved {len(datapoints)} DataPoints from {len(ontology_nodes)} nodes")
        return datapoints

    async def bind_to_graph(
        self,
        ontology: OntologyGraph,
        context: OntologyContext
    ) -> Tuple[List[Any], List[Any]]:  # (graph_nodes, graph_edges)
        """Bind ontology to graph structure using configured binding."""
        binding_config = self.domain_graph_bindings.get(context.domain)
        if not binding_config:
            logger.warning(f"No graph binding configured for domain: {context.domain}")
            # Use default binding
            binding_config = GraphBindingConfig()
        
        return await self.graph_binder.bind_ontology_to_graph(
            ontology, binding_config, context
        )

    def configure_datapoint_mapping(
        self,
        domain: str,
        mappings: List[DataPointMapping]
    ) -> None:
        """Configure DataPoint mappings for a domain."""
        self.domain_datapoint_mappings[domain] = mappings
        logger.info(f"Configured {len(mappings)} DataPoint mappings for domain: {domain}")

    def configure_graph_binding(
        self,
        domain: str,
        binding_config: GraphBindingConfig
    ) -> None:
        """Configure graph binding for a domain."""
        self.domain_graph_bindings[domain] = binding_config
        logger.info(f"Configured graph binding for domain: {domain}")

    def register_custom_resolver(
        self,
        resolver_name: str,
        resolver_func: Callable
    ) -> None:
        """Register a custom DataPoint resolver."""
        self.custom_resolvers[resolver_name] = resolver_func
        self.datapoint_resolver.register_custom_resolver(resolver_name, resolver_func)
        logger.info(f"Registered custom resolver: {resolver_name}")

    def register_binding_strategy(
        self,
        strategy_name: str,
        strategy_func: Callable
    ) -> None:
        """Register a custom graph binding strategy."""
        self.custom_binding_strategies[strategy_name] = strategy_func
        self.graph_binder.register_binding_strategy(strategy_name, strategy_func)
        logger.info(f"Registered custom binding strategy: {strategy_name}")

    def _prioritize_ontologies(self, ontologies: List[OntologyGraph]) -> List[OntologyGraph]:
        """Prioritize ontologies by scope (dataset > user > pipeline > domain > global)."""
        priority_order = [
            OntologyScope.DATASET,
            OntologyScope.USER, 
            OntologyScope.PIPELINE,
            OntologyScope.DOMAIN,
            OntologyScope.GLOBAL,
        ]
        
        seen_ids = set()
        prioritized = []
        
        for scope in priority_order:
            for ontology in ontologies:
                if ontology.scope == scope and ontology.id not in seen_ids:
                    prioritized.append(ontology)
                    seen_ids.add(ontology.id)
        
        return prioritized

    def _get_adapter_for_ontology(self, ontology: OntologyGraph) -> str:
        """Get the appropriate adapter name for an ontology."""
        # This could be made more sophisticated with adapter selection logic
        format_to_adapter = {
            "rdf_xml": "rdf_adapter",
            "owl": "rdf_adapter", 
            "json": "json_adapter",
            "llm_generated": "llm_adapter",
        }
        return format_to_adapter.get(ontology.format.value, "default_adapter")

    def _find_mapping_for_node(
        self, 
        node: OntologyNode, 
        mappings: List[DataPointMapping]
    ) -> Optional[DataPointMapping]:
        """Find the appropriate DataPoint mapping for a node."""
        for mapping in mappings:
            if mapping.ontology_node_type == node.type:
                return mapping
        return None


async def create_ontology_manager(
    registry: IOntologyRegistry,
    providers: Optional[Dict[str, IOntologyProvider]] = None,
    adapters: Optional[Dict[str, IOntologyAdapter]] = None,
    datapoint_resolver: Optional[IDataPointResolver] = None,
    graph_binder: Optional[IGraphBinder] = None,
) -> OntologyManager:
    """Factory function to create an OntologyManager with default implementations."""
    
    # Import default implementations
    from cognee.modules.ontology.registry import OntologyRegistry
    from cognee.modules.ontology.providers import RDFOntologyProvider, JSONOntologyProvider
    from cognee.modules.ontology.adapters import DefaultOntologyAdapter
    from cognee.modules.ontology.resolvers import DefaultDataPointResolver
    from cognee.modules.ontology.binders import DefaultGraphBinder
    
    if providers is None:
        providers = {
            "rdf_provider": RDFOntologyProvider(),
            "json_provider": JSONOntologyProvider(),
        }
    
    if adapters is None:
        adapters = {
            "default_adapter": DefaultOntologyAdapter(),
            "rdf_adapter": DefaultOntologyAdapter(),
            "json_adapter": DefaultOntologyAdapter(),
        }
    
    if datapoint_resolver is None:
        datapoint_resolver = DefaultDataPointResolver()
    
    if graph_binder is None:
        graph_binder = DefaultGraphBinder()
    
    return OntologyManager(
        registry=registry,
        providers=providers,
        adapters=adapters,
        datapoint_resolver=datapoint_resolver,
        graph_binder=graph_binder,
    )
