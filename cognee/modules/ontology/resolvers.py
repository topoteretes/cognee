"""DataPoint resolution implementations."""

import importlib
from typing import Any, Dict, List, Optional, Callable, Type
from uuid import uuid4

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IDataPointResolver,
    OntologyNode,
    DataPointMapping,
    OntologyContext,
)

logger = get_logger("DataPointResolver")


class DefaultDataPointResolver(IDataPointResolver):
    """Default implementation for DataPoint resolution."""

    def __init__(self):
        self.custom_resolvers: Dict[str, Callable] = {}

    async def resolve_to_datapoint(
        self,
        ontology_node: OntologyNode,
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> Any:  # DataPoint
        """Resolve ontology node to DataPoint instance."""
        
        # Use custom resolver if specified
        if mapping_config.custom_resolver and mapping_config.custom_resolver in self.custom_resolvers:
            return await self._apply_custom_resolver(
                ontology_node, mapping_config, context
            )
        
        # Use default resolution logic
        return await self._default_resolution(ontology_node, mapping_config, context)

    async def resolve_from_datapoint(
        self,
        datapoint: Any,  # DataPoint
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> OntologyNode:
        """Resolve DataPoint instance to ontology node."""
        
        # Extract properties from DataPoint
        datapoint_dict = datapoint.to_dict() if hasattr(datapoint, 'to_dict') else datapoint.__dict__
        
        # Map DataPoint fields to ontology properties
        ontology_properties = {}
        reverse_mappings = {v: k for k, v in mapping_config.field_mappings.items()}
        
        for datapoint_field, ontology_field in reverse_mappings.items():
            if datapoint_field in datapoint_dict:
                ontology_properties[ontology_field] = datapoint_dict[datapoint_field]
        
        # Create ontology node
        node = OntologyNode(
            id=str(datapoint.id) if hasattr(datapoint, 'id') else str(uuid4()),
            name=ontology_properties.get('name', str(datapoint.id)),
            type=mapping_config.ontology_node_type,
            description=ontology_properties.get('description', ''),
            category=ontology_properties.get('category', 'entity'),
            properties=ontology_properties
        )
        
        return node

    async def validate_mapping(
        self,
        mapping_config: DataPointMapping
    ) -> bool:
        """Validate mapping configuration."""
        try:
            # Check if DataPoint class exists
            module_path, class_name = mapping_config.datapoint_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            datapoint_class = getattr(module, class_name)
            
            # Validate field mappings
            if hasattr(datapoint_class, '__annotations__'):
                valid_fields = set(datapoint_class.__annotations__.keys())
                mapped_fields = set(mapping_config.field_mappings.values())
                
                invalid_fields = mapped_fields - valid_fields
                if invalid_fields:
                    logger.warning(f"Invalid field mappings: {invalid_fields}")
                    return False
            
            # Validate custom resolver if specified
            if mapping_config.custom_resolver:
                if mapping_config.custom_resolver not in self.custom_resolvers:
                    logger.warning(f"Custom resolver not found: {mapping_config.custom_resolver}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Mapping validation failed: {e}")
            return False

    def register_custom_resolver(
        self,
        resolver_name: str,
        resolver_func: Callable[[OntologyNode, DataPointMapping], Any]
    ) -> None:
        """Register a custom resolver function."""
        self.custom_resolvers[resolver_name] = resolver_func
        logger.info(f"Registered custom resolver: {resolver_name}")

    async def _apply_custom_resolver(
        self,
        ontology_node: OntologyNode,
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> Any:
        """Apply custom resolver function."""
        resolver_func = self.custom_resolvers[mapping_config.custom_resolver]
        
        if callable(resolver_func):
            try:
                # Call resolver function
                if context:
                    result = await resolver_func(ontology_node, mapping_config, context)
                else:
                    result = await resolver_func(ontology_node, mapping_config)
                return result
            except Exception as e:
                logger.error(f"Custom resolver failed: {e}")
                # Fallback to default resolution
                return await self._default_resolution(ontology_node, mapping_config, context)
        else:
            logger.error(f"Invalid custom resolver: {mapping_config.custom_resolver}")
            return await self._default_resolution(ontology_node, mapping_config, context)

    async def _default_resolution(
        self,
        ontology_node: OntologyNode,
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> Any:
        """Default resolution logic."""
        try:
            # Import the DataPoint class
            module_path, class_name = mapping_config.datapoint_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            datapoint_class = getattr(module, class_name)
            
            # Map ontology properties to DataPoint fields
            datapoint_data = {}
            
            # Apply field mappings
            for ontology_field, datapoint_field in mapping_config.field_mappings.items():
                if ontology_field in ontology_node.properties:
                    datapoint_data[datapoint_field] = ontology_node.properties[ontology_field]
                elif hasattr(ontology_node, ontology_field):
                    datapoint_data[datapoint_field] = getattr(ontology_node, ontology_field)
            
            # Set default mappings if not provided
            if 'id' not in datapoint_data:
                datapoint_data['id'] = ontology_node.id
            if 'type' not in datapoint_data:
                datapoint_data['type'] = ontology_node.type
            
            # Add ontology metadata
            if hasattr(datapoint_class, 'metadata'):
                datapoint_data['metadata'] = {
                    'type': ontology_node.type,
                    'index_fields': list(mapping_config.field_mappings.values()),
                    'ontology_source': True,
                    'ontology_node_id': ontology_node.id,
                }
            
            # Set ontology_valid flag
            datapoint_data['ontology_valid'] = True
            
            # Create DataPoint instance
            datapoint = datapoint_class(**datapoint_data)
            
            # Apply validation rules if specified
            if mapping_config.validation_rules:
                await self._apply_validation_rules(datapoint, mapping_config.validation_rules)
            
            return datapoint
            
        except Exception as e:
            logger.error(f"Default resolution failed for node {ontology_node.id}: {e}")
            return None

    async def _apply_validation_rules(
        self,
        datapoint: Any,
        validation_rules: List[str]
    ) -> None:
        """Apply validation rules to DataPoint."""
        for rule in validation_rules:
            try:
                # This is a simple implementation - in practice, you'd want 
                # a more sophisticated rule engine
                if rule.startswith("required:"):
                    field_name = rule.split(":", 1)[1]
                    if not hasattr(datapoint, field_name) or getattr(datapoint, field_name) is None:
                        raise ValueError(f"Required field {field_name} is missing")
                
                elif rule.startswith("type:"):
                    field_name, expected_type = rule.split(":", 2)[1:]
                    if hasattr(datapoint, field_name):
                        field_value = getattr(datapoint, field_name)
                        if field_value is not None and not isinstance(field_value, eval(expected_type)):
                            raise ValueError(f"Field {field_name} has wrong type")
                
                # Add more validation rules as needed
                
            except Exception as e:
                logger.warning(f"Validation rule '{rule}' failed: {e}")


class DomainSpecificResolver(DefaultDataPointResolver):
    """Domain-specific resolver with specialized logic."""

    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain

    async def _default_resolution(
        self,
        ontology_node: OntologyNode,
        mapping_config: DataPointMapping,
        context: Optional[OntologyContext] = None
    ) -> Any:
        """Domain-specific resolution logic."""
        
        # Apply domain-specific preprocessing
        if self.domain == "medical":
            ontology_node = await self._preprocess_medical_node(ontology_node)
        elif self.domain == "legal":
            ontology_node = await self._preprocess_legal_node(ontology_node)
        elif self.domain == "code":
            ontology_node = await self._preprocess_code_node(ontology_node)
        
        # Use parent's default resolution
        return await super()._default_resolution(ontology_node, mapping_config, context)

    async def _preprocess_medical_node(self, node: OntologyNode) -> OntologyNode:
        """Preprocess medical domain nodes."""
        # Add medical-specific property transformations
        if node.type == "Disease":
            node.properties["medical_category"] = "disease"
        elif node.type == "Symptom":
            node.properties["medical_category"] = "symptom"
        
        return node

    async def _preprocess_legal_node(self, node: OntologyNode) -> OntologyNode:
        """Preprocess legal domain nodes."""
        # Add legal-specific property transformations
        if node.type == "Law":
            node.properties["legal_category"] = "legislation"
        elif node.type == "Case":
            node.properties["legal_category"] = "precedent"
        
        return node

    async def _preprocess_code_node(self, node: OntologyNode) -> OntologyNode:
        """Preprocess code domain nodes."""
        # Add code-specific property transformations
        if node.type == "Function":
            node.properties["code_category"] = "function"
        elif node.type == "Class":
            node.properties["code_category"] = "class"
        
        return node
