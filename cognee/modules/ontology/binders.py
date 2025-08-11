"""Graph binding implementations for ontologies."""

from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import uuid4
from datetime import datetime, timezone

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IGraphBinder,
    OntologyGraph,
    OntologyNode,
    OntologyEdge,
    GraphBindingConfig,
    OntologyContext,
)

logger = get_logger("GraphBinder")


class DefaultGraphBinder(IGraphBinder):
    """Default implementation for binding ontology to graph structures."""

    def __init__(self):
        self.custom_strategies: Dict[str, Callable] = {}

    async def bind_ontology_to_graph(
        self,
        ontology: OntologyGraph,
        binding_config: GraphBindingConfig,
        context: Optional[OntologyContext] = None
    ) -> Tuple[List[Any], List[Any]]:  # (graph_nodes, graph_edges)
        """Bind ontology to graph structure."""
        
        # Use custom binding strategy if specified
        if binding_config.custom_binding_strategy and binding_config.custom_binding_strategy in self.custom_strategies:
            return await self._apply_custom_strategy(ontology, binding_config, context)
        
        # Use default binding logic
        return await self._default_binding(ontology, binding_config, context)

    async def transform_node_properties(
        self,
        node: OntologyNode,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Transform node properties according to binding config."""
        
        transformed_props = {}
        
        # Start with node's base properties
        base_props = {
            "id": node.id,
            "name": node.name,
            "type": node.type,
            "description": node.description,
            "category": node.category,
        }
        
        # Add custom properties
        base_props.update(node.properties)
        
        # Apply transformations
        for prop_name, prop_value in base_props.items():
            if prop_name in transformations:
                try:
                    transformed_props[prop_name] = transformations[prop_name](prop_value)
                except Exception as e:
                    logger.warning(f"Transformation failed for property {prop_name}: {e}")
                    transformed_props[prop_name] = prop_value
            else:
                transformed_props[prop_name] = prop_value
        
        # Add standard graph properties
        transformed_props.update({
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "ontology_source": True,
            "ontology_id": node.id,
        })
        
        return transformed_props

    async def transform_edge_properties(
        self,
        edge: OntologyEdge,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Transform edge properties according to binding config."""
        
        transformed_props = {}
        
        # Start with edge's base properties  
        base_props = {
            "source_node_id": edge.source_id,
            "target_node_id": edge.target_id,
            "relationship_name": edge.relationship_type,
            "weight": edge.weight,
        }
        
        # Add custom properties
        base_props.update(edge.properties)
        
        # Apply transformations
        for prop_name, prop_value in base_props.items():
            if prop_name in transformations:
                try:
                    transformed_props[prop_name] = transformations[prop_name](prop_value)
                except Exception as e:
                    logger.warning(f"Transformation failed for edge property {prop_name}: {e}")
                    transformed_props[prop_name] = prop_value
            else:
                transformed_props[prop_name] = prop_value
        
        # Add standard graph properties
        transformed_props.update({
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "ontology_source": True,
            "ontology_edge_id": edge.id,
        })
        
        return transformed_props

    def register_binding_strategy(
        self,
        strategy_name: str,
        strategy_func: Callable[[OntologyGraph, GraphBindingConfig], Tuple[List[Any], List[Any]]]
    ) -> None:
        """Register a custom binding strategy."""
        self.custom_strategies[strategy_name] = strategy_func
        logger.info(f"Registered custom binding strategy: {strategy_name}")

    async def _apply_custom_strategy(
        self,
        ontology: OntologyGraph,
        binding_config: GraphBindingConfig,
        context: Optional[OntologyContext] = None
    ) -> Tuple[List[Any], List[Any]]:
        """Apply custom binding strategy."""
        
        strategy_func = self.custom_strategies[binding_config.custom_binding_strategy]
        
        try:
            if context:
                return await strategy_func(ontology, binding_config, context)
            else:
                return await strategy_func(ontology, binding_config)
        except Exception as e:
            logger.error(f"Custom binding strategy failed: {e}")
            # Fallback to default binding
            return await self._default_binding(ontology, binding_config, context)

    async def _default_binding(
        self,
        ontology: OntologyGraph,
        binding_config: GraphBindingConfig,
        context: Optional[OntologyContext] = None
    ) -> Tuple[List[Any], List[Any]]:
        """Default binding logic."""
        
        graph_nodes = []
        graph_edges = []
        
        # Process nodes
        for node in ontology.nodes:
            try:
                graph_node = await self._bind_node_to_graph(node, binding_config)
                if graph_node:
                    graph_nodes.append(graph_node)
            except Exception as e:
                logger.error(f"Failed to bind node {node.id}: {e}")
        
        # Process edges
        for edge in ontology.edges:
            try:
                graph_edge = await self._bind_edge_to_graph(edge, binding_config)
                if graph_edge:
                    graph_edges.append(graph_edge)
            except Exception as e:
                logger.error(f"Failed to bind edge {edge.id}: {e}")
        
        logger.info(f"Bound {len(graph_nodes)} nodes and {len(graph_edges)} edges to graph")
        return graph_nodes, graph_edges

    async def _bind_node_to_graph(
        self,
        node: OntologyNode,
        binding_config: GraphBindingConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Bind a single node to graph format."""
        
        # Map node type if configured
        graph_node_type = binding_config.node_type_mapping.get(node.type, node.type)
        
        # Transform properties
        node_properties = await self.transform_node_properties(
            node, binding_config.property_transformations
        )
        
        # Set the mapped type
        node_properties["type"] = graph_node_type
        
        # Generate node ID for graph (use ontology ID as base)
        node_id = node.id
        
        return (node_id, node_properties)

    async def _bind_edge_to_graph(
        self,
        edge: OntologyEdge,
        binding_config: GraphBindingConfig
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """Bind a single edge to graph format."""
        
        # Map edge type if configured
        graph_edge_type = binding_config.edge_type_mapping.get(
            edge.relationship_type, edge.relationship_type
        )
        
        # Transform properties
        edge_properties = await self.transform_edge_properties(
            edge, binding_config.property_transformations
        )
        
        # Set the mapped relationship name
        edge_properties["relationship_name"] = graph_edge_type
        
        return (
            edge.source_id,
            edge.target_id,
            graph_edge_type,
            edge_properties
        )


class KnowledgeGraphBinder(DefaultGraphBinder):
    """Specialized binder for KnowledgeGraph format."""

    async def _bind_node_to_graph(
        self,
        node: OntologyNode,
        binding_config: GraphBindingConfig
    ) -> Dict[str, Any]:
        """Bind node to KnowledgeGraph Node format."""
        
        # Transform properties
        node_properties = await self.transform_node_properties(
            node, binding_config.property_transformations
        )
        
        # Create KnowledgeGraph-compatible node
        from cognee.shared.data_models import Node
        
        kg_node = Node(
            id=node.id,
            name=node.name,
            type=binding_config.node_type_mapping.get(node.type, node.type),
            description=node.description or "",
        )
        
        return kg_node

    async def _bind_edge_to_graph(
        self,
        edge: OntologyEdge,
        binding_config: GraphBindingConfig
    ) -> Dict[str, Any]:
        """Bind edge to KnowledgeGraph Edge format."""
        
        from cognee.shared.data_models import Edge
        
        # Map edge type
        relationship_name = binding_config.edge_type_mapping.get(
            edge.relationship_type, edge.relationship_type
        )
        
        kg_edge = Edge(
            source_node_id=edge.source_id,
            target_node_id=edge.target_id,
            relationship_name=relationship_name,
        )
        
        return kg_edge


class DataPointGraphBinder(DefaultGraphBinder):
    """Specialized binder for DataPoint-based graphs."""

    async def _bind_node_to_graph(
        self,
        node: OntologyNode,
        binding_config: GraphBindingConfig
    ) -> Any:  # DataPoint
        """Bind node to DataPoint instance."""
        
        from cognee.infrastructure.engine.models.DataPoint import DataPoint
        
        # Transform properties
        node_properties = await self.transform_node_properties(
            node, binding_config.property_transformations
        )
        
        # Create DataPoint instance
        datapoint = DataPoint(
            id=node.id,
            type=binding_config.node_type_mapping.get(node.type, node.type),
            ontology_valid=True,
            metadata={
                "type": node.type,
                "index_fields": ["name", "type"],
                "ontology_source": True,
            }
        )
        
        # Add custom attributes
        for prop_name, prop_value in node_properties.items():
            if not hasattr(datapoint, prop_name):
                setattr(datapoint, prop_name, prop_value)
        
        return datapoint


class DomainSpecificBinder(DefaultGraphBinder):
    """Domain-specific graph binder with specialized transformation logic."""

    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain

    async def transform_node_properties(
        self,
        node: OntologyNode,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Apply domain-specific node transformations."""
        
        # Apply parent transformations first
        props = await super().transform_node_properties(node, transformations)
        
        # Apply domain-specific transformations
        if self.domain == "medical":
            props = await self._apply_medical_node_transforms(node, props)
        elif self.domain == "legal":
            props = await self._apply_legal_node_transforms(node, props)
        elif self.domain == "code":
            props = await self._apply_code_node_transforms(node, props)
        
        return props

    async def transform_edge_properties(
        self,
        edge: OntologyEdge,
        transformations: Dict[str, Callable[[Any], Any]]
    ) -> Dict[str, Any]:
        """Apply domain-specific edge transformations."""
        
        # Apply parent transformations first
        props = await super().transform_edge_properties(edge, transformations)
        
        # Apply domain-specific transformations
        if self.domain == "medical":
            props = await self._apply_medical_edge_transforms(edge, props)
        elif self.domain == "legal":
            props = await self._apply_legal_edge_transforms(edge, props)
        elif self.domain == "code":
            props = await self._apply_code_edge_transforms(edge, props)
        
        return props

    async def _apply_medical_node_transforms(
        self, node: OntologyNode, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply medical domain node transformations."""
        if node.type == "Disease":
            props["medical_category"] = "pathology"
            props["severity_level"] = node.properties.get("severity", "unknown")
        elif node.type == "Symptom":
            props["medical_category"] = "clinical_sign"
            props["frequency"] = node.properties.get("frequency", "unknown")
        
        return props

    async def _apply_legal_node_transforms(
        self, node: OntologyNode, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply legal domain node transformations."""
        if node.type == "Law":
            props["legal_authority"] = node.properties.get("jurisdiction", "unknown")
            props["enforcement_level"] = node.properties.get("level", "federal")
        elif node.type == "Case":
            props["court_level"] = node.properties.get("court", "unknown")
            props["precedent_value"] = node.properties.get("binding", "persuasive")
        
        return props

    async def _apply_code_node_transforms(
        self, node: OntologyNode, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply code domain node transformations."""
        if node.type == "Function":
            props["complexity"] = len(node.properties.get("parameters", []))
            props["visibility"] = node.properties.get("access_modifier", "public")
        elif node.type == "Class":
            props["inheritance_depth"] = node.properties.get("depth", 0)
            props["method_count"] = len(node.properties.get("methods", []))
        
        return props

    async def _apply_medical_edge_transforms(
        self, edge: OntologyEdge, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply medical domain edge transformations."""
        if edge.relationship_type == "causes":
            props["causality_strength"] = edge.properties.get("strength", "unknown")
        elif edge.relationship_type == "treats":
            props["efficacy"] = edge.properties.get("effectiveness", "unknown")
        
        return props

    async def _apply_legal_edge_transforms(
        self, edge: OntologyEdge, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply legal domain edge transformations."""
        if edge.relationship_type == "cites":
            props["citation_type"] = edge.properties.get("type", "supporting")
        elif edge.relationship_type == "overrules":
            props["authority_level"] = edge.properties.get("level", "same")
        
        return props

    async def _apply_code_edge_transforms(
        self, edge: OntologyEdge, props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply code domain edge transformations."""
        if edge.relationship_type == "calls":
            props["call_frequency"] = edge.properties.get("frequency", 1)
        elif edge.relationship_type == "inherits":
            props["inheritance_type"] = edge.properties.get("type", "extends")
        
        return props
