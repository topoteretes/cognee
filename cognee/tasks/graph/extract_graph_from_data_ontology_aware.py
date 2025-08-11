"""
Enhanced graph extraction task with ontology awareness.

This demonstrates how to update existing tasks to use the new ontology system.
"""

import asyncio
from typing import Type, List, Optional, Any, Dict

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.shared.logging_utils import get_logger

# New ontology imports
from cognee.modules.ontology.interfaces import (
    IOntologyManager,
    OntologyContext,
    DataPointMapping,
    GraphBindingConfig,
)

logger = get_logger("extract_graph_from_data_ontology_aware")


async def extract_graph_from_data_ontology_aware(
    data_chunks: List[DocumentChunk],
    graph_model: Type[Any] = KnowledgeGraph,
    ontology_manager: Optional[IOntologyManager] = None,
    ontology_context: Optional[OntologyContext] = None,
    datapoint_mappings: Optional[List[DataPointMapping]] = None,
    graph_binding_config: Optional[GraphBindingConfig] = None,
    entity_extraction_enabled: bool = False,
    target_entity_types: Optional[List[str]] = None,
    enhanced_content: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[DocumentChunk]:
    """
    Enhanced graph extraction with ontology awareness.
    
    Args:
        data_chunks: Document chunks to process
        graph_model: Graph model type (KnowledgeGraph or custom)
        ontology_manager: Ontology manager instance (injected by pipeline)
        ontology_context: Ontology context (injected by pipeline)
        datapoint_mappings: DataPoint mappings (injected by pipeline)
        graph_binding_config: Graph binding configuration (injected by pipeline)
        entity_extraction_enabled: Whether to use ontology for entity extraction
        target_entity_types: Specific entity types to extract
        enhanced_content: Pre-enhanced content with ontological information
        **kwargs: Additional parameters
    
    Returns:
        Updated document chunks with extracted graph data
    """
    
    logger.info(f"Processing {len(data_chunks)} chunks with ontology awareness")
    
    # Check if ontology integration is available
    if ontology_manager and ontology_context:
        logger.info(f"Ontology integration enabled for domain: {ontology_context.domain}")
        return await _extract_with_ontology(
            data_chunks=data_chunks,
            graph_model=graph_model,
            ontology_manager=ontology_manager,
            ontology_context=ontology_context,
            datapoint_mappings=datapoint_mappings,
            graph_binding_config=graph_binding_config,
            entity_extraction_enabled=entity_extraction_enabled,
            target_entity_types=target_entity_types,
            enhanced_content=enhanced_content,
            **kwargs
        )
    else:
        logger.info("No ontology integration, using standard extraction")
        return await _extract_standard(data_chunks, graph_model, **kwargs)


async def _extract_with_ontology(
    data_chunks: List[DocumentChunk],
    graph_model: Type[Any],
    ontology_manager: IOntologyManager,
    ontology_context: OntologyContext,
    datapoint_mappings: Optional[List[DataPointMapping]] = None,
    graph_binding_config: Optional[GraphBindingConfig] = None,
    entity_extraction_enabled: bool = False,
    target_entity_types: Optional[List[str]] = None,
    enhanced_content: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[DocumentChunk]:
    """Extract graph data with ontology enhancement."""
    
    # Step 1: Get applicable ontologies
    ontologies = await ontology_manager.get_applicable_ontologies(ontology_context)
    logger.info(f"Found {len(ontologies)} applicable ontologies")
    
    if not ontologies:
        logger.warning("No applicable ontologies found, falling back to standard extraction")
        return await _extract_standard(data_chunks, graph_model, **kwargs)
    
    # Step 2: Enhance content with ontological information
    chunk_graphs = []
    enhanced_datapoints = []
    
    for chunk in data_chunks:
        # Enhance chunk content if not already done
        if enhanced_content:
            chunk_enhanced = enhanced_content
        else:
            chunk_enhanced = await ontology_manager.enhance_with_ontology(
                chunk.text, ontology_context
            )
        
        # Extract graph using enhanced information
        chunk_graph = await _extract_chunk_graph_with_ontology(
            chunk=chunk,
            enhanced_content=chunk_enhanced,
            graph_model=graph_model,
            ontology_manager=ontology_manager,
            ontology_context=ontology_context,
            target_entity_types=target_entity_types,
        )
        
        chunk_graphs.append(chunk_graph)
        
        # Convert ontological entities to DataPoints if mappings available
        if datapoint_mappings and chunk_enhanced.get('extracted_entities'):
            ontology_nodes = await _convert_entities_to_ontology_nodes(
                chunk_enhanced['extracted_entities'], ontologies[0]
            )
            
            if ontology_nodes:
                datapoints = await ontology_manager.resolve_to_datapoints(
                    ontology_nodes, ontology_context
                )
                enhanced_datapoints.extend(datapoints)
    
    # Step 3: Integrate with graph database using custom binding
    if graph_model is KnowledgeGraph:
        # Use standard integration with ontology-enhanced nodes/edges
        await _integrate_ontology_enhanced_graphs(
            data_chunks=data_chunks,
            chunk_graphs=chunk_graphs,
            enhanced_datapoints=enhanced_datapoints,
            ontology_manager=ontology_manager,
            ontology_context=ontology_context,
            graph_binding_config=graph_binding_config,
        )
    else:
        # Custom graph model - just attach graphs to chunks
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph
    
    logger.info(f"Completed ontology-aware graph extraction for {len(data_chunks)} chunks")
    return data_chunks


async def _extract_chunk_graph_with_ontology(
    chunk: DocumentChunk,
    enhanced_content: Dict[str, Any],
    graph_model: Type[Any],
    ontology_manager: IOntologyManager,
    ontology_context: OntologyContext,
    target_entity_types: Optional[List[str]] = None,
) -> Any:
    """Extract graph for a single chunk using ontological enhancement."""
    
    # Build entity-aware prompt for LLM
    extracted_entities = enhanced_content.get('extracted_entities', [])
    semantic_relationships = enhanced_content.get('semantic_relationships', [])
    
    # Filter entities by target types if specified
    if target_entity_types:
        extracted_entities = [
            entity for entity in extracted_entities 
            if entity.get('type') in target_entity_types
        ]
    
    # Create enhanced prompt with ontological context
    ontology_context_prompt = _build_ontology_context_prompt(
        extracted_entities, semantic_relationships
    )
    
    # Use LLM to extract graph with ontological guidance
    full_prompt = f"""
    {ontology_context_prompt}
    
    Text to analyze:
    {chunk.text}
    
    Extract entities and relationships, giving preference to the ontological entities 
    and relationships mentioned above when they appear in the text.
    """
    
    # Extract using LLM with ontological context
    chunk_graph = await LLMGateway.acreate_structured_output(
        full_prompt,
        "You are a knowledge graph extractor. Use the ontological context to guide your extraction.",
        graph_model
    )
    
    # Enhance extracted graph with ontological metadata
    if hasattr(chunk_graph, 'nodes'):
        for node in chunk_graph.nodes:
            # Add ontological metadata to nodes
            matching_entity = _find_matching_ontological_entity(node, extracted_entities)
            if matching_entity:
                node.ontology_source = matching_entity.get('ontology_id')
                node.ontology_confidence = matching_entity.get('confidence', 0.0)
                if hasattr(node, 'type'):
                    node.type = matching_entity.get('type', node.type)
    
    return chunk_graph


async def _integrate_ontology_enhanced_graphs(
    data_chunks: List[DocumentChunk],
    chunk_graphs: List[Any],
    enhanced_datapoints: List[Any],
    ontology_manager: IOntologyManager,
    ontology_context: OntologyContext,
    graph_binding_config: Optional[GraphBindingConfig] = None,
):
    """Integrate ontology-enhanced graphs into the graph database."""
    
    graph_engine = await get_graph_engine()
    
    # Get existing edges to avoid duplicates
    existing_edges_map = await retrieve_existing_edges(data_chunks, chunk_graphs)
    
    # Apply ontology-aware graph binding if available
    if graph_binding_config:
        # Transform graphs using custom binding
        all_nodes = []
        all_edges = []
        
        for chunk_graph in chunk_graphs:
            if hasattr(chunk_graph, 'nodes') and hasattr(chunk_graph, 'edges'):
                # Convert chunk graph to ontology format for binding
                from cognee.modules.ontology.interfaces import OntologyGraph, OntologyNode, OntologyEdge
                
                ontology_nodes = []
                ontology_edges = []
                
                for node in chunk_graph.nodes:
                    ont_node = OntologyNode(
                        id=node.id,
                        name=node.name,
                        type=node.type,
                        description=getattr(node, 'description', ''),
                        properties=getattr(node, '__dict__', {})
                    )
                    ontology_nodes.append(ont_node)
                
                for edge in chunk_graph.edges:
                    ont_edge = OntologyEdge(
                        id=f"{edge.source_node_id}_{edge.target_node_id}_{edge.relationship_name}",
                        source_id=edge.source_node_id,
                        target_id=edge.target_node_id,
                        relationship_type=edge.relationship_name,
                        properties=getattr(edge, '__dict__', {})
                    )
                    ontology_edges.append(ont_edge)
                
                # Create temporary ontology for binding
                temp_ontology = OntologyGraph(
                    id="temp_chunk_ontology",
                    name="Chunk Ontology",
                    description="Temporary ontology for chunk graph",
                    format="llm_generated",
                    scope="dataset",
                    nodes=ontology_nodes,
                    edges=ontology_edges
                )
                
                # Apply custom binding
                bound_nodes, bound_edges = await ontology_manager.bind_to_graph(
                    temp_ontology, ontology_context
                )
                
                all_nodes.extend(bound_nodes)
                all_edges.extend(bound_edges)
        
        # Add bound nodes and edges to graph
        if all_nodes:
            await add_data_points(all_nodes)
        if all_edges:
            await graph_engine.add_edges(all_edges)
    
    else:
        # Use standard integration
        graph_nodes, graph_edges = expand_with_nodes_and_edges(
            data_chunks, chunk_graphs, None, existing_edges_map
        )
        
        if graph_nodes:
            await add_data_points(graph_nodes)
        if graph_edges:
            await graph_engine.add_edges(graph_edges)
    
    # Add enhanced DataPoints from ontology resolution
    if enhanced_datapoints:
        await add_data_points(enhanced_datapoints)
        logger.info(f"Added {len(enhanced_datapoints)} ontology-resolved DataPoints")


async def _extract_standard(
    data_chunks: List[DocumentChunk],
    graph_model: Type[Any],
    **kwargs
) -> List[DocumentChunk]:
    """Standard graph extraction without ontology (fallback)."""
    
    # Import the original function to maintain compatibility
    from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
    
    # Generate standard graphs using LLM
    chunk_graphs = []
    for chunk in data_chunks:
        system_prompt = LLMGateway.read_query_prompt("generate_graph_prompt_oneshot.txt")
        chunk_graph = await LLMGateway.acreate_structured_output(
            chunk.text, system_prompt, graph_model
        )
        chunk_graphs.append(chunk_graph)
    
    # Use standard integration
    return await integrate_chunk_graphs(
        data_chunks=data_chunks,
        chunk_graphs=chunk_graphs,
        graph_model=graph_model,
        ontology_adapter=None,  # No ontology adapter
    )


def _build_ontology_context_prompt(
    extracted_entities: List[Dict[str, Any]],
    semantic_relationships: List[Dict[str, Any]]
) -> str:
    """Build ontological context prompt for LLM."""
    
    if not extracted_entities and not semantic_relationships:
        return ""
    
    prompt = "ONTOLOGICAL CONTEXT:\n"
    
    if extracted_entities:
        prompt += "Known entities in this domain:\n"
        for entity in extracted_entities[:10]:  # Limit to top 10
            prompt += f"- {entity['name']} (type: {entity['type']})\n"
        prompt += "\n"
    
    if semantic_relationships:
        prompt += "Known relationships:\n"
        for rel in semantic_relationships[:10]:  # Limit to top 10
            prompt += f"- {rel['source']} {rel['relationship']} {rel['target']}\n"
        prompt += "\n"
    
    prompt += "When extracting entities and relationships, prefer these ontological concepts when they appear in the text.\n\n"
    
    return prompt


def _find_matching_ontological_entity(
    extracted_node: Any, 
    ontological_entities: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find matching ontological entity for an extracted node."""
    
    node_name = getattr(extracted_node, 'name', '').lower()
    
    for entity in ontological_entities:
        entity_name = entity.get('name', '').lower()
        if node_name == entity_name or node_name in entity_name or entity_name in node_name:
            return entity
    
    return None


async def _convert_entities_to_ontology_nodes(
    extracted_entities: List[Dict[str, Any]],
    ontology: Any
) -> List[Any]:
    """Convert extracted entities to ontology nodes for DataPoint resolution."""
    
    from cognee.modules.ontology.interfaces import OntologyNode
    
    ontology_nodes = []
    
    for entity in extracted_entities:
        node = OntologyNode(
            id=entity.get('node_id', entity['name']),
            name=entity['name'],
            type=entity['type'],
            description=entity.get('description', ''),
            category=entity.get('category', 'entity'),
            properties={
                'confidence': entity.get('confidence', 0.0),
                'ontology_id': entity.get('ontology_id'),
                'source': 'llm_extraction'
            }
        )
        ontology_nodes.append(node)
    
    return ontology_nodes
