"""Configuration system for ontology integration."""

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import yaml

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    DataPointMapping,
    GraphBindingConfig,
    OntologyFormat,
    OntologyScope,
)
from cognee.modules.ontology.pipeline_integration import (
    create_medical_pipeline_config,
    create_legal_pipeline_config,
    create_code_pipeline_config,
)

logger = get_logger("OntologyConfiguration")


class OntologyConfiguration:
    """Central configuration management for ontology system."""

    def __init__(self):
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        self.pipeline_configs: Dict[str, Dict[str, Any]] = {}
        self.custom_resolvers: Dict[str, Callable] = {}
        self.custom_binding_strategies: Dict[str, Callable] = {}
        
        # Load default configurations
        self._load_default_configs()

    def _load_default_configs(self):
        """Load default domain configurations."""
        self.domain_configs.update({
            "medical": create_medical_pipeline_config(),
            "legal": create_legal_pipeline_config(),
            "code": create_code_pipeline_config(),
        })

    def register_domain_config(
        self,
        domain: str,
        datapoint_mappings: List[DataPointMapping],
        graph_binding: GraphBindingConfig,
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Register configuration for a domain."""
        
        self.domain_configs[domain] = {
            "domain": domain,
            "datapoint_mappings": datapoint_mappings,
            "graph_binding": graph_binding,
            "task_configs": task_configs or {},
        }
        
        logger.info(f"Registered domain configuration: {domain}")

    def register_pipeline_config(
        self,
        pipeline_name: str,
        domain: str,
        custom_mappings: Optional[List[DataPointMapping]] = None,
        custom_binding: Optional[GraphBindingConfig] = None,
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Register configuration for a specific pipeline."""
        
        # Start with domain config if available
        base_config = self.domain_configs.get(domain, {})
        
        pipeline_config = {
            "pipeline_name": pipeline_name,
            "domain": domain,
            "datapoint_mappings": custom_mappings or base_config.get("datapoint_mappings", []),
            "graph_binding": custom_binding or base_config.get("graph_binding", GraphBindingConfig()),
            "task_configs": {**base_config.get("task_configs", {}), **(task_configs or {})},
        }
        
        self.pipeline_configs[pipeline_name] = pipeline_config
        logger.info(f"Registered pipeline configuration: {pipeline_name}")

    def get_domain_config(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a domain."""
        return self.domain_configs.get(domain)

    def get_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a pipeline."""
        return self.pipeline_configs.get(pipeline_name)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from file."""
        
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        self._parse_config_data(config_data)
        logger.info(f"Loaded configuration from {config_file}")

    def save_to_file(self, config_file: str, format: str = "json") -> None:
        """Save configuration to file."""
        
        config_data = {
            "domains": self._serialize_domain_configs(),
            "pipelines": self._serialize_pipeline_configs(),
        }
        
        config_path = Path(config_file)
        
        if format.lower() == "json":
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        elif format.lower() in ["yml", "yaml"]:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {config_file}")

    def register_custom_resolver(
        self,
        resolver_name: str,
        resolver_func: Callable
    ) -> None:
        """Register a custom DataPoint resolver."""
        self.custom_resolvers[resolver_name] = resolver_func
        logger.info(f"Registered custom resolver: {resolver_name}")

    def register_custom_binding_strategy(
        self,
        strategy_name: str,
        strategy_func: Callable
    ) -> None:
        """Register a custom graph binding strategy."""
        self.custom_binding_strategies[strategy_name] = strategy_func
        logger.info(f"Registered custom binding strategy: {strategy_name}")

    def create_datapoint_mapping(
        self,
        ontology_node_type: str,
        datapoint_class: str,
        field_mappings: Optional[Dict[str, str]] = None,
        custom_resolver: Optional[str] = None,
        validation_rules: Optional[List[str]] = None
    ) -> DataPointMapping:
        """Create a DataPointMapping configuration."""
        
        return DataPointMapping(
            ontology_node_type=ontology_node_type,
            datapoint_class=datapoint_class,
            field_mappings=field_mappings or {},
            custom_resolver=custom_resolver,
            validation_rules=validation_rules or []
        )

    def create_graph_binding_config(
        self,
        node_type_mapping: Optional[Dict[str, str]] = None,
        edge_type_mapping: Optional[Dict[str, str]] = None,
        property_transformations: Optional[Dict[str, Callable]] = None,
        custom_binding_strategy: Optional[str] = None
    ) -> GraphBindingConfig:
        """Create a GraphBindingConfig."""
        
        return GraphBindingConfig(
            node_type_mapping=node_type_mapping or {},
            edge_type_mapping=edge_type_mapping or {},
            property_transformations=property_transformations or {},
            custom_binding_strategy=custom_binding_strategy
        )

    def _parse_config_data(self, config_data: Dict[str, Any]) -> None:
        """Parse configuration data from file."""
        
        # Parse domain configurations
        for domain, domain_config in config_data.get("domains", {}).items():
            mappings = []
            for mapping_data in domain_config.get("datapoint_mappings", []):
                mapping = DataPointMapping(**mapping_data)
                mappings.append(mapping)
            
            binding_data = domain_config.get("graph_binding", {})
            binding = GraphBindingConfig(**binding_data)
            
            task_configs = domain_config.get("task_configs", {})
            
            self.register_domain_config(domain, mappings, binding, task_configs)
        
        # Parse pipeline configurations
        for pipeline_name, pipeline_config in config_data.get("pipelines", {}).items():
            domain = pipeline_config.get("domain")
            
            custom_mappings = None
            if "datapoint_mappings" in pipeline_config:
                custom_mappings = []
                for mapping_data in pipeline_config["datapoint_mappings"]:
                    mapping = DataPointMapping(**mapping_data)
                    custom_mappings.append(mapping)
            
            custom_binding = None
            if "graph_binding" in pipeline_config:
                binding_data = pipeline_config["graph_binding"]
                custom_binding = GraphBindingConfig(**binding_data)
            
            task_configs = pipeline_config.get("task_configs", {})
            
            self.register_pipeline_config(
                pipeline_name, domain, custom_mappings, custom_binding, task_configs
            )

    def _serialize_domain_configs(self) -> Dict[str, Any]:
        """Serialize domain configurations for saving."""
        serialized = {}
        
        for domain, config in self.domain_configs.items():
            serialized[domain] = {
                "domain": config["domain"],
                "datapoint_mappings": [
                    mapping.dict() for mapping in config["datapoint_mappings"]
                ],
                "graph_binding": config["graph_binding"].dict(),
                "task_configs": config["task_configs"],
            }
        
        return serialized

    def _serialize_pipeline_configs(self) -> Dict[str, Any]:
        """Serialize pipeline configurations for saving."""
        serialized = {}
        
        for pipeline_name, config in self.pipeline_configs.items():
            serialized[pipeline_name] = {
                "pipeline_name": config["pipeline_name"],
                "domain": config["domain"],
                "datapoint_mappings": [
                    mapping.dict() for mapping in config["datapoint_mappings"]
                ],
                "graph_binding": config["graph_binding"].dict(),
                "task_configs": config["task_configs"],
            }
        
        return serialized


# Global configuration instance
_global_ontology_config = None


def get_ontology_config() -> OntologyConfiguration:
    """Get the global ontology configuration instance."""
    global _global_ontology_config
    if _global_ontology_config is None:
        _global_ontology_config = OntologyConfiguration()
    return _global_ontology_config


def configure_domain(
    domain: str,
    datapoint_mappings: List[DataPointMapping],
    graph_binding: GraphBindingConfig,
    task_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """Configure ontology for a domain (convenience function)."""
    config = get_ontology_config()
    config.register_domain_config(domain, datapoint_mappings, graph_binding, task_configs)


def configure_pipeline(
    pipeline_name: str,
    domain: str,
    custom_mappings: Optional[List[DataPointMapping]] = None,
    custom_binding: Optional[GraphBindingConfig] = None,
    task_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """Configure ontology for a pipeline (convenience function)."""
    config = get_ontology_config()
    config.register_pipeline_config(
        pipeline_name, domain, custom_mappings, custom_binding, task_configs
    )


def load_ontology_config(config_file: str) -> None:
    """Load ontology configuration from file (convenience function)."""
    config = get_ontology_config()
    config.load_from_file(config_file)


# Example configuration templates
def create_example_config_file(output_file: str) -> None:
    """Create an example configuration file."""
    
    example_config = {
        "domains": {
            "example_domain": {
                "domain": "example_domain",
                "datapoint_mappings": [
                    {
                        "ontology_node_type": "Entity",
                        "datapoint_class": "cognee.infrastructure.engine.models.DataPoint.DataPoint",
                        "field_mappings": {
                            "name": "name",
                            "description": "description",
                            "category": "entity_type"
                        },
                        "custom_resolver": None,
                        "validation_rules": ["required:name"]
                    }
                ],
                "graph_binding": {
                    "node_type_mapping": {
                        "Entity": "domain_entity",
                        "Concept": "domain_concept"
                    },
                    "edge_type_mapping": {
                        "related_to": "domain_relation",
                        "part_of": "composition"
                    },
                    "property_transformations": {},
                    "custom_binding_strategy": None
                },
                "task_configs": {
                    "extract_graph_from_data": {
                        "enhance_with_entities": True,
                        "inject_datapoint_mappings": True,
                        "inject_graph_binding": True,
                        "target_entity_types": ["Entity", "Concept"]
                    }
                }
            }
        },
        "pipelines": {
            "example_pipeline": {
                "pipeline_name": "example_pipeline",
                "domain": "example_domain",
                "datapoint_mappings": [],  # Use domain defaults
                "graph_binding": {},       # Use domain defaults
                "task_configs": {
                    "summarize_text": {
                        "enhance_with_entities": True,
                        "enable_ontology_validation": True
                    }
                }
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    logger.info(f"Created example configuration file: {output_file}")
