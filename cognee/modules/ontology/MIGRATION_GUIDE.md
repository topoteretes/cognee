# Ontology System Migration Guide

This guide explains how to migrate from the old ontology system to the new refactored architecture.

## Overview of Changes

The ontology system has been completely refactored to provide:

1. **Better Separation of Concerns**: Clear interfaces for different components
2. **DataPoint Integration**: Automatic mapping between ontologies and DataPoint instances
3. **Custom Graph Binding**: Configurable binding strategies for different graph types
4. **Pipeline Integration**: Seamless injection into pipeline tasks
5. **Domain Configuration**: Pre-configured setups for common domains
6. **Extensibility**: Plugin system for custom behavior

## Architecture Changes

### Old System
```
OntologyResolver (monolithic)
├── RDF/OWL parsing
├── Basic node/edge extraction
└── Simple graph operations
```

### New System
```
OntologyManager (orchestrator)
├── OntologyRegistry (storage/lookup)
├── OntologyProviders (format-specific loading)
├── OntologyAdapters (query/search operations)
├── DataPointResolver (ontology ↔ DataPoint mapping)
└── GraphBinder (ontology → graph structure binding)
```

## Migration Steps

### 1. Replace OntologyResolver Usage

**Old Code:**
```python
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver

# Old way
ontology_resolver = OntologyResolver(ontology_file="medical.owl")
nodes, edges, root = ontology_resolver.get_subgraph("Disease", "classes")
```

**New Code:**
```python
from cognee.modules.ontology import (
    create_ontology_system,
    OntologyContext,
    configure_domain,
    DataPointMapping,
    GraphBindingConfig
)

# New way
ontology_manager = await create_ontology_system()

# Configure domain if needed
mappings = [
    DataPointMapping(
        ontology_node_type="Disease",
        datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
        field_mappings={"name": "name", "description": "description"}
    )
]
binding = GraphBindingConfig(
    node_type_mapping={"Disease": "medical_condition"}
)
configure_domain("medical", mappings, binding)

# Use with context
context = OntologyContext(domain="medical")
ontologies = await ontology_manager.get_applicable_ontologies(context)
```

### 2. Update Pipeline Task Integration

**Old Code:**
```python
from cognee.api.v1.cognify.cognify import get_default_tasks

# Old way - hardcoded ontology adapter
tasks = await get_default_tasks(
    ontology_file_path="path/to/ontology.owl"
)
```

**New Code:**
```python
from cognee.modules.ontology import create_pipeline_injector
from cognee.modules.pipelines.tasks.task import Task

# New way - configurable ontology injection
ontology_manager = await create_ontology_system()
injector = create_pipeline_injector(ontology_manager, "my_pipeline", "medical")

# Inject into tasks
original_task = Task(extract_graph_from_data)
context = OntologyContext(domain="medical", pipeline_name="my_pipeline")
enhanced_task = await injector.inject_into_task(original_task, context)
```

### 3. Update DataPoint Creation

**Old Code:**
```python
# Old way - manual DataPoint creation
from cognee.infrastructure.engine.models.DataPoint import DataPoint

datapoint = DataPoint(
    id="disease_001",
    type="Disease", 
    # Manual field mapping...
)
```

**New Code:**
```python
# New way - automatic resolution from ontology
ontology_nodes = [...]  # Nodes from ontology
context = OntologyContext(domain="medical")

# Automatically resolve to DataPoints
datapoints = await ontology_manager.resolve_to_datapoints(ontology_nodes, context)
```

### 4. Update Graph Binding

**Old Code:**
```python
# Old way - hardcoded graph node creation
graph_nodes = []
for ontology_node in nodes:
    graph_node = (
        ontology_node.id,
        {
            "name": ontology_node.name,
            "type": ontology_node.type,
            # Manual property mapping...
        }
    )
    graph_nodes.append(graph_node)
```

**New Code:**
```python
# New way - configurable binding
ontology = # ... loaded ontology
context = OntologyContext(domain="medical")

# Automatically bind to graph structure
graph_nodes, graph_edges = await ontology_manager.bind_to_graph(ontology, context)
```

## Domain-Specific Configurations

### Medical Domain

**Old Code:**
```python
# Old way - manual setup for each pipeline
ontology_resolver = OntologyResolver("medical_ontology.owl")
# ... manual entity extraction
# ... manual graph binding
```

**New Code:**
```python
# New way - use pre-configured medical domain
from cognee.modules.ontology import create_medical_pipeline_config

ontology_manager = await create_ontology_system()
injector = create_pipeline_injector(ontology_manager, "medical_pipeline", "medical")

# Automatically configured for:
# - Disease, Symptom, Treatment entities
# - Medical-specific DataPoint mappings
# - Clinical graph relationships
```

### Legal Domain

```python
# Pre-configured for legal documents
injector = create_pipeline_injector(ontology_manager, "legal_pipeline", "legal")

# Automatically handles:
# - Law, Case, Court entities
# - Legal citation relationships
# - Jurisdiction-aware processing
```

### Code Analysis Domain

```python
# Pre-configured for code analysis
injector = create_pipeline_injector(ontology_manager, "code_pipeline", "code")

# Automatically handles:
# - Function, Class, Module entities
# - Code dependency relationships
# - Language-specific processing
```

## Custom Resolvers and Binding

### Custom DataPoint Resolver

```python
# Define custom resolver for special entity types
async def custom_medical_resolver(ontology_node, mapping_config, context=None):
    # Custom logic for creating DataPoints
    datapoint = DataPoint(
        id=ontology_node.id,
        type="medical_entity",
        # Custom field mapping and validation
    )
    return datapoint

# Register the resolver
ontology_manager.register_custom_resolver("medical_resolver", custom_medical_resolver)

# Use in mapping configuration
mapping = DataPointMapping(
    ontology_node_type="SpecialDisease",
    datapoint_class="cognee.infrastructure.engine.models.DataPoint.DataPoint",
    custom_resolver="medical_resolver"
)
```

### Custom Graph Binding

```python
# Define custom binding strategy
async def custom_graph_binding(ontology, binding_config, context=None):
    # Custom logic for graph structure creation
    graph_nodes = []
    graph_edges = []
    
    for node in ontology.nodes:
        # Custom node transformation
        transformed_node = transform_node(node)
        graph_nodes.append(transformed_node)
    
    return graph_nodes, graph_edges

# Register the strategy
ontology_manager.register_binding_strategy("custom_binding", custom_graph_binding)

# Use in binding configuration
binding = GraphBindingConfig(
    custom_binding_strategy="custom_binding"
)
```

## Configuration Files

### Create Configuration File

```python
from cognee.modules.ontology import create_example_config_file

# Generate example configuration
create_example_config_file("ontology_config.json")
```

### Load Configuration

```python
from cognee.modules.ontology import load_ontology_config

# Load from file
load_ontology_config("ontology_config.json")

# Or create system with config
ontology_manager = await create_ontology_system(config_file="ontology_config.json")
```

## Backward Compatibility

The old `OntologyResolver` is still available for backward compatibility:

```python
from cognee.modules.ontology import OntologyResolver

# Old interface still works
resolver = OntologyResolver(ontology_file="medical.owl")
```

However, it's recommended to migrate to the new system for better flexibility and features.

## Performance Improvements

The new system provides several performance benefits:

1. **Lazy Loading**: Ontologies are loaded only when needed
2. **Caching**: Registry caches frequently accessed ontologies
3. **Parallel Processing**: Multiple ontologies can be processed simultaneously
4. **Semantic Search**: Optional semantic similarity for better entity matching

## Testing Your Migration

Use the example usage script to test your migration:

```python
from cognee.modules.ontology.example_usage import main

# Run comprehensive examples
await main()
```

## Common Migration Issues

### 1. Import Errors

**Problem**: `ImportError: cannot import name 'OntologyResolver'`

**Solution**: Update imports to use new interfaces:
```python
# Old
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver

# New  
from cognee.modules.ontology import create_ontology_system, OntologyContext
```

### 2. Task Parameter Changes

**Problem**: Tasks expecting `ontology_adapter` parameter

**Solution**: Use ontology injection:
```python
# Old
Task(extract_graph_from_data, ontology_adapter=resolver)

# New
injector = create_pipeline_injector(ontology_manager, "pipeline", "domain")
enhanced_task = await injector.inject_into_task(original_task, context)
```

### 3. DataPoint Creation Changes

**Problem**: Manual DataPoint creation with ontology data

**Solution**: Use automatic resolution:
```python
# Old
datapoint = DataPoint(id=node.id, type=node.type, ...)

# New
datapoints = await ontology_manager.resolve_to_datapoints(nodes, context)
```

## Support

For additional support with migration:

1. Check the example usage file: `cognee/modules/ontology/example_usage.py`
2. Review the interface documentation in `cognee/modules/ontology/interfaces.py` 
3. Use the pre-configured domain setups for common use cases
4. Test with the provided configuration examples

The new system is designed to be more powerful while maintaining ease of use. Most migrations can be completed by updating imports and using the convenience functions provided.
