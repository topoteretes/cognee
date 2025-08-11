"""
Example usage of the refactored ontology system.

This demonstrates how to:
1. Set up ontology providers and registry
2. Configure domain-specific mappings
3. Integrate with pipelines
4. Use custom resolvers and binding strategies
"""

import asyncio
from pathlib import Path
from typing import List, Any, Dict

from cognee.modules.ontology.interfaces import (
    OntologyContext,
    OntologyScope,
    DataPointMapping,
    GraphBindingConfig,
)
from cognee.modules.ontology.manager import create_ontology_manager
from cognee.modules.ontology.registry import OntologyRegistry
from cognee.modules.ontology.providers import JSONOntologyProvider, RDFOntologyProvider
from cognee.modules.ontology.adapters import DefaultOntologyAdapter
from cognee.modules.ontology.resolvers import DefaultDataPointResolver, DomainSpecificResolver
from cognee.modules.ontology.binders import DefaultGraphBinder, DomainSpecificBinder
from cognee.modules.ontology.pipeline_integration import (
    PipelineOntologyConfigurator,
    OntologyInjector,
)
from cognee.modules.ontology.configuration import (
    get_ontology_config,
    configure_domain,
    configure_pipeline,
)
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger

logger = get_logger("OntologyExample")


async def example_basic_setup():
    """Example: Basic ontology system setup."""
    
    logger.info("=== Basic Ontology System Setup ===")
    
    # 1. Create registry and providers
    registry = OntologyRegistry()
    
    providers = {
        "json_provider": JSONOntologyProvider(),
        "rdf_provider": RDFOntologyProvider(),
    }
    
    adapters = {
        "default_adapter": DefaultOntologyAdapter(),
    }
    
    # 2. Create resolvers and binders
    datapoint_resolver = DefaultDataPointResolver()
    graph_binder = DefaultGraphBinder()
    
    # 3. Create ontology manager
    ontology_manager = await create_ontology_manager(
        registry=registry,
        providers=providers,
        adapters=adapters,
        datapoint_resolver=datapoint_resolver,
        graph_binder=graph_binder,
    )
    
    logger.info("Ontology system initialized successfully")
    return ontology_manager


async def example_load_ontologies(ontology_manager):
    """Example: Loading different types of ontologies."""
    
    logger.info("=== Loading Ontologies ===")
    
    # 1. Load JSON ontology
    json_ontology_data = {
        "id": "medical_ontology",
        "name": "Medical Knowledge Base",
        "description": "Basic medical ontology",
        "nodes": [
            {
                "id": "disease_001",
                "name": "Diabetes",
                "type": "Disease",
                "description": "A group of metabolic disorders",
                "category": "medical_condition",
                "properties": {"icd_code": "E11", "severity": "chronic"}
            },
            {
                "id": "symptom_001", 
                "name": "Fatigue",
                "type": "Symptom",
                "description": "Extreme tiredness",
                "category": "clinical_finding",
                "properties": {"frequency": "common"}
            }
        ],
        "edges": [
            {
                "id": "rel_001",
                "source": "disease_001",
                "target": "symptom_001", 
                "relationship": "causes",
                "properties": {"strength": "moderate"}
            }
        ]
    }
    
    json_provider = ontology_manager.providers["json_provider"]
    medical_ontology = await json_provider.load_ontology(json_ontology_data)
    
    # 2. Register ontology in registry
    ontology_id = await ontology_manager.registry.register_ontology(
        medical_ontology,
        OntologyScope.DOMAIN,
        OntologyContext(domain="medical")
    )
    
    logger.info(f"Loaded and registered medical ontology: {ontology_id}")
    
    return medical_ontology


async def example_configure_domain_mappings(ontology_manager):
    """Example: Configure domain-specific DataPoint mappings."""
    
    logger.info("=== Configuring Domain Mappings ===")
    
    # 1. Define DataPoint mappings for medical domain
    medical_mappings = [
        DataPointMapping(
            ontology_node_type="Disease",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description", 
                "icd_code": "medical_code",
                "severity": "severity_level",
            },
            validation_rules=["required:name", "required:medical_code"]
        ),
        DataPointMapping(
            ontology_node_type="Symptom",
            datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
            field_mappings={
                "name": "name",
                "description": "description",
                "frequency": "occurrence_rate",
            }
        )
    ]
    
    # 2. Define graph binding configuration
    medical_binding = GraphBindingConfig(
        node_type_mapping={
            "Disease": "medical_condition",
            "Symptom": "clinical_finding",
            "Treatment": "therapeutic_procedure",
        },
        edge_type_mapping={
            "causes": "causality",
            "treats": "therapeutic_relationship",
            "associated_with": "clinical_association",
        }
    )
    
    # 3. Configure the domain
    ontology_manager.configure_datapoint_mapping("medical", medical_mappings)
    ontology_manager.configure_graph_binding("medical", medical_binding)
    
    logger.info("Configured medical domain mappings and bindings")


async def example_custom_resolver(ontology_manager):
    """Example: Register and use custom DataPoint resolver."""
    
    logger.info("=== Custom DataPoint Resolver ===")
    
    # 1. Define custom resolver function
    async def medical_disease_resolver(ontology_node, mapping_config, context=None):
        """Custom resolver for medical disease entities."""
        from cognee.infrastructure.engine.models.DataPoint import DataPoint
        
        # Create DataPoint with medical-specific logic
        datapoint = DataPoint(
            id=ontology_node.id,
            type="medical_disease",
            ontology_valid=True,
            metadata={
                "type": "medical_entity",
                "index_fields": ["name", "medical_code"],
                "domain": "medical",
                "ontology_node_id": ontology_node.id,
            }
        )
        
        # Map ontology properties with domain-specific processing
        datapoint.name = ontology_node.name
        datapoint.description = ontology_node.description
        datapoint.medical_code = ontology_node.properties.get("icd_code", "")
        datapoint.severity_level = ontology_node.properties.get("severity", "unknown")
        
        # Add computed properties
        datapoint.medical_category = "disease"
        datapoint.risk_level = "high" if datapoint.severity_level == "chronic" else "low"
        
        logger.info(f"Custom resolver created DataPoint for disease: {datapoint.name}")
        return datapoint
    
    # 2. Register custom resolver
    ontology_manager.register_custom_resolver("medical_disease_resolver", medical_disease_resolver)
    
    # 3. Update mapping to use custom resolver
    updated_mapping = DataPointMapping(
        ontology_node_type="Disease",
        datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
        field_mappings={},  # Handled by custom resolver
        custom_resolver="medical_disease_resolver"
    )
    
    ontology_manager.configure_datapoint_mapping("medical", [updated_mapping])
    
    logger.info("Registered custom medical disease resolver")


async def example_custom_binding_strategy(ontology_manager):
    """Example: Register and use custom graph binding strategy."""
    
    logger.info("=== Custom Graph Binding Strategy ===")
    
    # 1. Define custom binding strategy
    async def medical_graph_binding_strategy(ontology, binding_config, context=None):
        """Custom binding strategy for medical graphs."""
        from datetime import datetime
        
        graph_nodes = []
        graph_edges = []
        
        # Process nodes with medical-specific transformations
        for node in ontology.nodes:
            if node.type == "Disease":
                # Create medical condition node
                node_props = {
                    "id": node.id,
                    "name": node.name,
                    "type": "medical_condition",
                    "description": node.description,
                    "medical_code": node.properties.get("icd_code", ""),
                    "severity": node.properties.get("severity", "unknown"),
                    "category": "pathology",
                    "updated_at": datetime.now().isoformat(),
                    "ontology_source": True,
                }
                graph_nodes.append((node.id, node_props))
            
            elif node.type == "Symptom":
                # Create clinical finding node
                node_props = {
                    "id": node.id,
                    "name": node.name,
                    "type": "clinical_finding",
                    "description": node.description,
                    "frequency": node.properties.get("frequency", "unknown"),
                    "category": "symptom",
                    "updated_at": datetime.now().isoformat(),
                    "ontology_source": True,
                }
                graph_nodes.append((node.id, node_props))
        
        # Process edges with medical-specific relationships
        for edge in ontology.edges:
            edge_props = {
                "source_node_id": edge.source_id,
                "target_node_id": edge.target_id,
                "relationship_name": "medical_" + edge.relationship_type,
                "strength": edge.properties.get("strength", "unknown"),
                "updated_at": datetime.now().isoformat(),
                "ontology_source": True,
            }
            
            graph_edges.append((
                edge.source_id,
                edge.target_id,
                "medical_" + edge.relationship_type,
                edge_props
            ))
        
        logger.info(f"Custom binding strategy created {len(graph_nodes)} nodes and {len(graph_edges)} edges")
        return graph_nodes, graph_edges
    
    # 2. Register custom binding strategy
    ontology_manager.register_binding_strategy("medical_graph_binding", medical_graph_binding_strategy)
    
    # 3. Update binding config to use custom strategy
    updated_binding = GraphBindingConfig(
        custom_binding_strategy="medical_graph_binding"
    )
    
    ontology_manager.configure_graph_binding("medical", updated_binding)
    
    logger.info("Registered custom medical graph binding strategy")


async def example_pipeline_integration(ontology_manager):
    """Example: Integrate ontology with pipeline tasks."""
    
    logger.info("=== Pipeline Integration ===")
    
    # 1. Create pipeline configurator
    pipeline_configurator = PipelineOntologyConfigurator(ontology_manager)
    
    # 2. Configure medical pipeline
    medical_mappings = ontology_manager.domain_datapoint_mappings.get("medical", [])
    medical_binding = ontology_manager.domain_graph_bindings.get("medical")
    
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
    
    pipeline_configurator.configure_pipeline(
        pipeline_name="medical_cognify_pipeline",
        domain="medical",
        datapoint_mappings=medical_mappings,
        graph_binding=medical_binding,
        task_specific_configs=task_configs
    )
    
    # 3. Create ontology injector for the pipeline
    injector = pipeline_configurator.create_ontology_injector("medical_cognify_pipeline")
    
    # 4. Create sample task and inject ontology
    async def sample_extract_task(data_chunks, **kwargs):
        """Sample extraction task."""
        logger.info("Executing extract task with ontology context")
        ontology_context = kwargs.get("ontology_context")
        datapoint_mappings = kwargs.get("datapoint_mappings", [])
        
        logger.info(f"Task received ontology context for domain: {ontology_context.domain}")
        logger.info(f"Task has {len(datapoint_mappings)} DataPoint mappings available")
        
        # Simulate task processing with ontology enhancement
        enhanced_results = []
        for chunk in data_chunks:
            # In real implementation, this would use ontology for entity extraction
            enhanced_results.append({
                "chunk": chunk,
                "ontology_enhanced": True,
                "domain": ontology_context.domain,
                "entities_found": ["Diabetes", "Fatigue"]  # Simulated
            })
        
        return enhanced_results
    
    sample_task = Task(sample_extract_task)
    
    # 5. Get pipeline context and inject ontology
    context = pipeline_configurator.get_pipeline_context(
        "medical_cognify_pipeline",
        user_id="user123",
        dataset_id="medical_dataset_001"
    )
    
    enhanced_task = await injector.inject_into_task(sample_task, context)
    
    # 6. Execute enhanced task
    sample_data = ["Patient reports fatigue and frequent urination", "Diagnosis: Type 2 Diabetes"]
    results = await enhanced_task.run(sample_data)
    
    logger.info(f"Enhanced task completed with {len(results)} results")
    
    return results


async def example_content_enhancement(ontology_manager):
    """Example: Enhance content with ontological information."""
    
    logger.info("=== Content Enhancement ===")
    
    # 1. Create context for medical domain
    context = OntologyContext(
        domain="medical",
        pipeline_name="medical_analysis",
        user_id="doctor123"
    )
    
    # 2. Sample medical text
    medical_text = """
    The patient presents with chronic fatigue and excessive thirst. 
    Blood glucose levels are elevated, indicating possible diabetes mellitus.
    Further testing for HbA1c is recommended to confirm diagnosis.
    """
    
    # 3. Enhance content with ontology
    enhanced_content = await ontology_manager.enhance_with_ontology(medical_text, context)
    
    logger.info("Enhanced content:")
    logger.info(f"  Original content: {enhanced_content['original_content'][:50]}...")
    logger.info(f"  Extracted entities: {len(enhanced_content['extracted_entities'])}")
    logger.info(f"  Semantic relationships: {len(enhanced_content['semantic_relationships'])}")
    
    for entity in enhanced_content['extracted_entities']:
        logger.info(f"    Found entity: {entity['name']} (type: {entity['type']})")
    
    for relationship in enhanced_content['semantic_relationships']:
        logger.info(f"    Relationship: {relationship['source']} -> {relationship['relationship']} -> {relationship['target']}")
    
    return enhanced_content


async def example_datapoint_resolution(ontology_manager):
    """Example: Resolve ontology nodes to DataPoint instances."""
    
    logger.info("=== DataPoint Resolution ===")
    
    # 1. Get medical ontology
    context = OntologyContext(domain="medical")
    ontologies = await ontology_manager.get_applicable_ontologies(context)
    
    if not ontologies:
        logger.warning("No medical ontologies found")
        return []
    
    medical_ontology = ontologies[0]
    
    # 2. Filter disease nodes
    disease_nodes = [node for node in medical_ontology.nodes if node.type == "Disease"]
    
    if not disease_nodes:
        logger.warning("No disease nodes found in ontology")
        return []
    
    # 3. Resolve to DataPoint instances
    datapoints = await ontology_manager.resolve_to_datapoints(disease_nodes, context)
    
    logger.info(f"Resolved {len(datapoints)} disease nodes to DataPoints:")
    for dp in datapoints:
        logger.info(f"  DataPoint: {dp.type} - {getattr(dp, 'name', 'Unnamed')}")
        logger.info(f"    ID: {dp.id}")
        logger.info(f"    Ontology valid: {dp.ontology_valid}")
        if hasattr(dp, 'medical_code'):
            logger.info(f"    Medical code: {dp.medical_code}")
    
    return datapoints


async def example_graph_binding(ontology_manager):
    """Example: Bind ontology to graph structure."""
    
    logger.info("=== Graph Binding ===")
    
    # 1. Get medical ontology
    context = OntologyContext(domain="medical")
    ontologies = await ontology_manager.get_applicable_ontologies(context)
    
    if not ontologies:
        logger.warning("No medical ontologies found")
        return [], []
    
    medical_ontology = ontologies[0]
    
    # 2. Bind to graph structure
    graph_nodes, graph_edges = await ontology_manager.bind_to_graph(medical_ontology, context)
    
    logger.info(f"Bound ontology to graph structure:")
    logger.info(f"  Graph nodes: {len(graph_nodes)}")
    logger.info(f"  Graph edges: {len(graph_edges)}")
    
    # Display sample nodes
    for i, node in enumerate(graph_nodes[:3]):  # Show first 3
        if isinstance(node, tuple):
            node_id, node_props = node
            logger.info(f"    Node {i+1}: {node_id} (type: {node_props.get('type', 'unknown')})")
        else:
            logger.info(f"    Node {i+1}: {getattr(node, 'id', 'unknown')} (type: {getattr(node, 'type', 'unknown')})")
    
    # Display sample edges
    for i, edge in enumerate(graph_edges[:3]):  # Show first 3
        if isinstance(edge, tuple) and len(edge) >= 4:
            source, target, rel_type, props = edge[:4]
            logger.info(f"    Edge {i+1}: {source} -> {rel_type} -> {target}")
        else:
            logger.info(f"    Edge {i+1}: {edge}")
    
    return graph_nodes, graph_edges


async def main():
    """Run all examples."""
    
    logger.info("Starting ontology system examples...")
    
    try:
        # 1. Basic setup
        ontology_manager = await example_basic_setup()
        
        # 2. Load ontologies
        medical_ontology = await example_load_ontologies(ontology_manager)
        
        # 3. Configure domain mappings
        await example_configure_domain_mappings(ontology_manager)
        
        # 4. Custom resolver
        await example_custom_resolver(ontology_manager)
        
        # 5. Custom binding strategy
        await example_custom_binding_strategy(ontology_manager)
        
        # 6. Pipeline integration
        pipeline_results = await example_pipeline_integration(ontology_manager)
        
        # 7. Content enhancement
        enhanced_content = await example_content_enhancement(ontology_manager)
        
        # 8. DataPoint resolution
        datapoints = await example_datapoint_resolution(ontology_manager)
        
        # 9. Graph binding
        graph_nodes, graph_edges = await example_graph_binding(ontology_manager)
        
        logger.info("All examples completed successfully!")
        
        # Print summary
        logger.info("\n=== Summary ===")
        logger.info(f"Loaded ontologies: 1")
        logger.info(f"Pipeline results: {len(pipeline_results) if pipeline_results else 0}")
        logger.info(f"Enhanced entities: {len(enhanced_content.get('extracted_entities', [])) if enhanced_content else 0}")
        logger.info(f"DataPoints created: {len(datapoints) if datapoints else 0}")
        logger.info(f"Graph nodes: {len(graph_nodes) if graph_nodes else 0}")
        logger.info(f"Graph edges: {len(graph_edges) if graph_edges else 0}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
