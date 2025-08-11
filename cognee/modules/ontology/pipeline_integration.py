"""Pipeline integration for ontology system."""

from typing import Any, Dict, List, Optional, Type, Union
import inspect

from cognee.shared.logging_utils import get_logger
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.ontology.interfaces import (
    IOntologyManager,
    OntologyContext,
    DataPointMapping,
    GraphBindingConfig,
)

logger = get_logger("OntologyPipelineIntegration")


class OntologyInjector:
    """Handles injection of ontology context into pipeline tasks."""

    def __init__(self, ontology_manager: IOntologyManager):
        self.ontology_manager = ontology_manager
        self.task_ontology_configs: Dict[str, Dict[str, Any]] = {}

    def configure_task_ontology(
        self,
        task_name: str,
        ontology_config: Dict[str, Any]
    ) -> None:
        """Configure ontology settings for a specific task."""
        self.task_ontology_configs[task_name] = ontology_config
        logger.info(f"Configured ontology for task: {task_name}")

    async def inject_into_task(
        self,
        task: Task,
        context: OntologyContext
    ) -> Task:
        """Inject ontology context into a task."""
        
        task_name = self._get_task_name(task)
        
        # Check if task has ontology configuration
        if task_name not in self.task_ontology_configs:
            # No specific configuration, use default behavior
            return await self._apply_default_injection(task, context)
        
        # Apply configured ontology injection
        config = self.task_ontology_configs[task_name]
        return await self._apply_configured_injection(task, context, config)

    async def enhance_task_params(
        self,
        task_params: Dict[str, Any],
        context: OntologyContext,
        task_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhance task parameters with ontological context."""
        
        enhanced_params = await self.ontology_manager.inject_ontology_into_task(
            task_name or "unknown_task",
            task_params,
            context
        )
        
        return enhanced_params

    def _get_task_name(self, task: Task) -> str:
        """Extract task name from Task object."""
        if hasattr(task.executable, '__name__'):
            return task.executable.__name__
        elif hasattr(task.executable, '__class__'):
            return task.executable.__class__.__name__
        else:
            return str(task.executable)

    async def _apply_default_injection(
        self,
        task: Task,
        context: OntologyContext
    ) -> Task:
        """Apply default ontology injection to task."""
        
        # Get applicable ontologies
        ontologies = await self.ontology_manager.get_applicable_ontologies(context)
        
        if not ontologies:
            logger.debug("No applicable ontologies found for task")
            return task
        
        # Enhance task parameters
        enhanced_params = task.default_params.copy()
        enhanced_params["kwargs"]["ontology_context"] = context
        enhanced_params["kwargs"]["available_ontologies"] = [ont.id for ont in ontologies]
        
        # Create new task with enhanced parameters
        enhanced_task = Task(
            task.executable,
            *enhanced_params["args"],
            task_config=task.task_config,
            **enhanced_params["kwargs"]
        )
        
        return enhanced_task

    async def _apply_configured_injection(
        self,
        task: Task,
        context: OntologyContext,
        config: Dict[str, Any]
    ) -> Task:
        """Apply configured ontology injection to task."""
        
        enhanced_params = task.default_params.copy()
        
        # Apply ontology-specific enhancements based on config
        if config.get("enhance_with_entities", False):
            enhanced_params = await self._inject_entity_enhancement(
                enhanced_params, context, config
            )
        
        if config.get("inject_datapoint_mappings", False):
            enhanced_params = await self._inject_datapoint_mappings(
                enhanced_params, context, config
            )
        
        if config.get("inject_graph_binding", False):
            enhanced_params = await self._inject_graph_binding(
                enhanced_params, context, config
            )
        
        if config.get("enable_ontology_validation", False):
            enhanced_params = await self._inject_validation_config(
                enhanced_params, context, config
            )
        
        # Create new task with enhanced parameters
        enhanced_task = Task(
            task.executable,
            *enhanced_params["args"],
            task_config=task.task_config,
            **enhanced_params["kwargs"]
        )
        
        return enhanced_task

    async def _inject_entity_enhancement(
        self,
        params: Dict[str, Any],
        context: OntologyContext,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject entity enhancement capabilities."""
        
        params["kwargs"]["ontology_manager"] = self.ontology_manager
        params["kwargs"]["ontology_context"] = context
        params["kwargs"]["entity_extraction_enabled"] = True
        
        # Add specific entity types to extract if configured
        if "target_entity_types" in config:
            params["kwargs"]["target_entity_types"] = config["target_entity_types"]
        
        return params

    async def _inject_datapoint_mappings(
        self,
        params: Dict[str, Any],
        context: OntologyContext,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject DataPoint mapping configurations."""
        
        # Get domain-specific mappings
        if context.domain:
            domain_mappings = getattr(
                self.ontology_manager, 'domain_datapoint_mappings', {}
            ).get(context.domain, [])
            
            if domain_mappings:
                params["kwargs"]["datapoint_mappings"] = domain_mappings
                params["kwargs"]["datapoint_resolver"] = self.ontology_manager.datapoint_resolver
        
        # Add custom mappings from config
        if "custom_mappings" in config:
            params["kwargs"]["custom_datapoint_mappings"] = config["custom_mappings"]
        
        return params

    async def _inject_graph_binding(
        self,
        params: Dict[str, Any],
        context: OntologyContext,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject graph binding configurations."""
        
        # Get domain-specific binding config
        if context.domain:
            domain_binding = getattr(
                self.ontology_manager, 'domain_graph_bindings', {}
            ).get(context.domain)
            
            if domain_binding:
                params["kwargs"]["graph_binding_config"] = domain_binding
                params["kwargs"]["graph_binder"] = self.ontology_manager.graph_binder
        
        # Add custom binding from config
        if "custom_binding" in config:
            params["kwargs"]["custom_graph_binding"] = config["custom_binding"]
        
        return params

    async def _inject_validation_config(
        self,
        params: Dict[str, Any],
        context: OntologyContext,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject ontology validation configurations."""
        
        params["kwargs"]["ontology_validation_enabled"] = True
        params["kwargs"]["validation_threshold"] = config.get("validation_threshold", 0.8)
        params["kwargs"]["strict_validation"] = config.get("strict_validation", False)
        
        return params


class OntologyAwareTaskWrapper:
    """Wrapper to make existing tasks ontology-aware."""

    def __init__(
        self,
        original_task: Task,
        ontology_manager: IOntologyManager,
        context: OntologyContext
    ):
        self.original_task = original_task
        self.ontology_manager = ontology_manager
        self.context = context

    async def execute_with_ontology(self, *args, **kwargs):
        """Execute task with ontology enhancements."""
        
        # Enhance content if provided
        if "content" in kwargs:
            enhanced_content = await self.ontology_manager.enhance_with_ontology(
                kwargs["content"], self.context
            )
            kwargs["enhanced_content"] = enhanced_content
        
        # Add ontology context
        kwargs["ontology_context"] = self.context
        kwargs["ontology_manager"] = self.ontology_manager
        
        # Execute original task
        return await self.original_task.run(*args, **kwargs)


class PipelineOntologyConfigurator:
    """Configures ontology integration for entire pipelines."""

    def __init__(self, ontology_manager: IOntologyManager):
        self.ontology_manager = ontology_manager
        self.pipeline_configs: Dict[str, Dict[str, Any]] = {}

    def configure_pipeline(
        self,
        pipeline_name: str,
        domain: str,
        datapoint_mappings: List[DataPointMapping],
        graph_binding: GraphBindingConfig,
        task_specific_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Configure ontology for an entire pipeline."""
        
        # Configure domain mappings
        self.ontology_manager.configure_datapoint_mapping(domain, datapoint_mappings)
        self.ontology_manager.configure_graph_binding(domain, graph_binding)
        
        # Store pipeline configuration
        self.pipeline_configs[pipeline_name] = {
            "domain": domain,
            "datapoint_mappings": datapoint_mappings,
            "graph_binding": graph_binding,
            "task_configs": task_specific_configs or {},
        }
        
        logger.info(f"Configured ontology for pipeline: {pipeline_name}")

    def get_pipeline_context(
        self,
        pipeline_name: str,
        user_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        custom_properties: Optional[Dict[str, Any]] = None
    ) -> OntologyContext:
        """Get ontology context for a pipeline."""
        
        config = self.pipeline_configs.get(pipeline_name, {})
        
        return OntologyContext(
            user_id=user_id,
            dataset_id=dataset_id,
            pipeline_name=pipeline_name,
            domain=config.get("domain"),
            custom_properties=custom_properties or {}
        )

    def create_ontology_injector(self, pipeline_name: str) -> OntologyInjector:
        """Create an ontology injector configured for a specific pipeline."""
        
        injector = OntologyInjector(self.ontology_manager)
        
        # Apply pipeline-specific task configurations
        if pipeline_name in self.pipeline_configs:
            task_configs = self.pipeline_configs[pipeline_name].get("task_configs", {})
            for task_name, config in task_configs.items():
                injector.configure_task_ontology(task_name, config)
        
        return injector


# Pre-configured pipeline setups for common domains
def create_medical_pipeline_config() -> Dict[str, Any]:
    """Create pre-configured ontology setup for medical pipelines."""
    
    datapoint_mappings = [
        DataPointMapping(
            ontology_node_type="Disease",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "icd_code": "medical_code",
                "severity": "severity_level",
            },
            validation_rules=["required:name", "type:severity_level:str"]
        ),
        DataPointMapping(
            ontology_node_type="Symptom",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "frequency": "occurrence_rate",
            }
        ),
    ]
    
    graph_binding = GraphBindingConfig(
        node_type_mapping={
            "Disease": "medical_entity",
            "Symptom": "clinical_finding",
            "Treatment": "therapeutic_procedure",
        },
        edge_type_mapping={
            "treats": "therapeutic_relationship",
            "causes": "causality",
            "associated_with": "clinical_association",
        }
    )
    
    task_configs = {
        "extract_graph_from_data": {
            "enhance_with_entities": True,
            "inject_datapoint_mappings": True,
            "inject_graph_binding": True,
            "target_entity_types": ["Disease", "Symptom", "Treatment"],
        },
        "summarize_text": {
            "enhance_with_entities": True,
            "enable_ontology_validation": True,
            "validation_threshold": 0.85,
        }
    }
    
    return {
        "domain": "medical",
        "datapoint_mappings": datapoint_mappings,
        "graph_binding": graph_binding,
        "task_configs": task_configs,
    }


def create_legal_pipeline_config() -> Dict[str, Any]:
    """Create pre-configured ontology setup for legal pipelines."""
    
    datapoint_mappings = [
        DataPointMapping(
            ontology_node_type="Law",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "jurisdiction": "legal_authority",
                "citation": "legal_citation",
            }
        ),
        DataPointMapping(
            ontology_node_type="Case",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "court": "court_level",
                "date": "decision_date",
            }
        ),
    ]
    
    graph_binding = GraphBindingConfig(
        node_type_mapping={
            "Law": "legal_statute",
            "Case": "legal_precedent",
            "Court": "judicial_body",
        },
        edge_type_mapping={
            "cites": "legal_citation",
            "overrules": "legal_override",
            "applies": "legal_application",
        }
    )
    
    task_configs = {
        "extract_graph_from_data": {
            "enhance_with_entities": True,
            "inject_datapoint_mappings": True,
            "inject_graph_binding": True,
            "target_entity_types": ["Law", "Case", "Court", "Legal_Concept"],
        }
    }
    
    return {
        "domain": "legal",
        "datapoint_mappings": datapoint_mappings,
        "graph_binding": graph_binding,
        "task_configs": task_configs,
    }


def create_code_pipeline_config() -> Dict[str, Any]:
    """Create pre-configured ontology setup for code analysis pipelines."""
    
    datapoint_mappings = [
        DataPointMapping(
            ontology_node_type="Function",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "parameters": "function_parameters",
                "return_type": "return_type",
            }
        ),
        DataPointMapping(
            ontology_node_type="Class",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "methods": "class_methods",
                "inheritance": "parent_classes",
            }
        ),
    ]
    
    graph_binding = GraphBindingConfig(
        node_type_mapping={
            "Function": "code_function",
            "Class": "code_class",
            "Module": "code_module",
            "Variable": "code_variable",
        },
        edge_type_mapping={
            "calls": "function_call",
            "inherits": "inheritance",
            "imports": "module_import",
            "defines": "definition",
        }
    )
    
    task_configs = {
        "extract_graph_from_code": {
            "enhance_with_entities": True,
            "inject_datapoint_mappings": True,
            "inject_graph_binding": True,
            "target_entity_types": ["Function", "Class", "Module", "Variable"],
        }
    }
    
    return {
        "domain": "code",
        "datapoint_mappings": datapoint_mappings,
        "graph_binding": graph_binding,
        "task_configs": task_configs,
    }
