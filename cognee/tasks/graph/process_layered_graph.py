"""
Tasks for processing layered knowledge graphs.

This module provides tasks that can be used in Cognee's pipeline system to process
layered knowledge graphs, including storing them in graph databases and extracting
information from them.
"""

import asyncio
import logging
from uuid import UUID
from typing import List, Dict, Any, Optional, Union, Type

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP,
)
from cognee.modules.graph.layered_graph_adapter import LayeredGraphDBAdapter
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph

logger = logging.getLogger(__name__)


async def extract_layered_graph_from_content(
    content: str,
    layer_config: Optional[List[Dict[str, Any]]] = None,
    base_model: Type[BaseModel] = KnowledgeGraph,
) -> LayeredKnowledgeGraphDP:
    """
    Extract a layered knowledge graph from content.

    Args:
        content: Text content to extract the graph from
        layer_config: Optional configuration for layers.
                     Each item should be a dict with 'name', 'description', and 'prompt' keys.
                     If not provided, a single layer will be created.
        base_model: The base model to use for graph extraction

    Returns:
        A LayeredKnowledgeGraphDP instance
    """
    # Create an empty layered graph
    layered_graph = LayeredKnowledgeGraphDP.create_empty(
        name="Extracted Layered Graph", description="Graph extracted from content with layers"
    )

    if not layer_config:
        # Default to a single layer if no config is provided
        layer_config = [
            {
                "name": "Base Layer",
                "description": "Basic information extracted from content",
                "prompt": "Extract the main entities and relationships from the content",
            }
        ]

    # Process each layer according to configuration
    parent_layers = []
    for layer_idx, layer_conf in enumerate(layer_config):
        # Create a layer
        layer = GraphLayer.create(
            name=layer_conf.get("name", f"Layer {layer_idx}"),
            description=layer_conf.get("description", f"Layer {layer_idx} description"),
            layer_type=layer_conf.get("layer_type", "default"),
            parent_layers=parent_layers.copy() if layer_idx > 0 else [],
        )

        # Add layer to graph
        layered_graph.add_layer(layer)

        # Extract knowledge graph for this layer with specific prompt
        prompt = layer_conf.get("prompt", "")
        if prompt:
            # Use the prompt to guide extraction
            extraction_prompt = f"{prompt}\n\nContent: {content}"
        else:
            extraction_prompt = content

        # Extract graph using the base model
        graph = await extract_content_graph(extraction_prompt, base_model)

        # Convert nodes and edges to GraphNode and GraphEdge and add to layer
        for node in graph.nodes:
            graph_node = GraphNode.from_basic_node(node, layer.id)
            layered_graph.add_node(graph_node, layer.id)

        for edge in graph.edges:
            # We need to find the corresponding source and target nodes
            source_id = None
            target_id = None

            # Find source node
            for node_id, node in layered_graph.nodes.items():
                if node.layer_id == layer.id and node.name == edge.source_node_id:
                    source_id = node_id
                    break

            # Find target node
            for node_id, node in layered_graph.nodes.items():
                if node.layer_id == layer.id and node.name == edge.target_node_id:
                    target_id = node_id
                    break

            if source_id and target_id:
                graph_edge = GraphEdge.create(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relationship_name=edge.relationship_name,
                    layer_id=layer.id,
                )
                layered_graph.add_edge(graph_edge, layer.id)

        # Add current layer to parent layers for the next iteration
        parent_layers.append(layer.id)

    return layered_graph


async def extract_layered_graph_from_data(
    data_points: List[DataPoint],
    layer_config: Optional[List[Dict[str, Any]]] = None,
    base_model: Type[BaseModel] = KnowledgeGraph,
) -> List[DataPoint]:
    """
    Extract layered knowledge graphs from data points and add them to the original data points.

    Args:
        data_points: List of data points containing text to extract graphs from
        layer_config: Optional configuration for layers
        base_model: The base model to use for graph extraction

    Returns:
        The original data points with layered graphs added
    """
    # Process each data point
    for data_point in data_points:
        # Get the text to process from the data point
        text = None
        for field_name in ["text", "content", "description"]:
            if hasattr(data_point, field_name):
                text = getattr(data_point, field_name)
                if text:
                    break

        if not text:
            logger.warning(f"No text found in data point {data_point.id}")
            continue

        # Extract layered graph
        layered_graph = await extract_layered_graph_from_content(text, layer_config, base_model)

        # Add the layered graph to the data point
        setattr(data_point, "layered_graph", layered_graph)

    return data_points


async def store_layered_graphs(data_points: List[DataPoint]) -> List[DataPoint]:
    """
    Store layered knowledge graphs attached to data points in the graph database.

    Args:
        data_points: List of data points with layered graphs attached

    Returns:
        The original data points
    """
    adapter = LayeredGraphDBAdapter()

    # Process each data point
    for data_point in data_points:
        if hasattr(data_point, "layered_graph") and isinstance(
            data_point.layered_graph, LayeredKnowledgeGraphDP
        ):
            # Store the layered graph
            graph_id = await adapter.store_graph(data_point.layered_graph)

            # Add the graph ID to the data point
            setattr(data_point, "layered_graph_id", graph_id)

    return data_points


async def retrieve_layered_graphs(data_points: List[DataPoint]) -> List[DataPoint]:
    """
    Retrieve layered knowledge graphs for data points that have a graph ID but no graph object.

    Args:
        data_points: List of data points

    Returns:
        The data points with retrieved graphs
    """
    adapter = LayeredGraphDBAdapter()

    # Process each data point
    for data_point in data_points:
        if hasattr(data_point, "layered_graph_id") and not hasattr(data_point, "layered_graph"):
            # Retrieve the layered graph
            graph = await adapter.retrieve_graph(data_point.layered_graph_id)

            # Add the graph to the data point
            setattr(data_point, "layered_graph", graph)

    return data_points


async def enrich_layered_graph(
    graph: LayeredKnowledgeGraphDP,
    enrichment_type: str,
    content: Optional[str] = None,
    parent_layer_ids: Optional[List[UUID]] = None,
) -> LayeredKnowledgeGraphDP:
    """
    Add an enrichment layer to a layered knowledge graph.

    Args:
        graph: The layered knowledge graph to enrich
        enrichment_type: Type of enrichment (e.g., "classification", "summarization")
        content: Optional additional content to consider for enrichment
        parent_layer_ids: Optional list of parent layer IDs. If None, all existing layers will be parents.

    Returns:
        The enriched layered knowledge graph
    """
    # Determine parent layers
    if parent_layer_ids is None:
        parent_layer_ids = list(graph.layers.keys())

    # Create an enrichment layer
    layer = GraphLayer.create(
        name=f"{enrichment_type.capitalize()} Layer",
        description=f"Layer with {enrichment_type} enrichments",
        layer_type="enrichment",
        parent_layers=parent_layer_ids,
        properties={"enrichment_type": enrichment_type},
    )

    # Add layer to graph
    graph.add_layer(layer)

    # Get existing graph content
    existing_content = ""
    for layer_id in parent_layer_ids:
        layer_graph = graph.get_layer_graph(layer_id)
        for node in layer_graph.nodes:
            existing_content += f"Node: {node.name} ({node.type}): {node.description}\n"
        for edge in layer_graph.edges:
            existing_content += f"Relationship: {edge.source_node_id} --[{edge.relationship_name}]--> {edge.target_node_id}\n"

    # Combine with additional content if provided
    if content:
        combined_content = f"{existing_content}\n\nAdditional content: {content}"
    else:
        combined_content = existing_content

    # Create prompt based on enrichment type
    if enrichment_type == "classification":
        prompt = f"Classify the entities in the following content and add new classification nodes: {combined_content}"
    elif enrichment_type == "summarization":
        prompt = f"Create summary nodes for the key concepts in the following content: {combined_content}"
    elif enrichment_type == "inference":
        prompt = f"Infer additional relationships and entities from the following content: {combined_content}"
    else:
        prompt = f"Enrich the following content with additional information: {combined_content}"

    # Extract enrichment graph
    enrichment_graph = await extract_content_graph(prompt, KnowledgeGraph)

    # Add nodes to the enrichment layer
    for node in enrichment_graph.nodes:
        graph_node = GraphNode.from_basic_node(node, layer.id)
        graph.add_node(graph_node, layer.id)

    # Add edges to the enrichment layer
    for edge in enrichment_graph.edges:
        # We need to find the corresponding source and target nodes
        source_id = None
        target_id = None

        # First try to find in the enrichment layer
        for node_id, node in graph.nodes.items():
            if node.layer_id == layer.id:
                if node.name == edge.source_node_id:
                    source_id = node_id
                elif node.name == edge.target_node_id:
                    target_id = node_id

        # If not found, look in parent layers
        if not source_id or not target_id:
            for node_id, node in graph.nodes.items():
                if node.layer_id in parent_layer_ids:
                    if not source_id and node.name == edge.source_node_id:
                        source_id = node_id
                    elif not target_id and node.name == edge.target_node_id:
                        target_id = node_id

        # Create the edge if both nodes were found
        if source_id and target_id:
            graph_edge = GraphEdge.create(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_name=edge.relationship_name,
                layer_id=layer.id,
            )
            graph.add_edge(graph_edge, layer.id)

    return graph


async def process_layered_graphs_with_pipeline(
    data_points: List[DataPoint], pipeline_config: List[Dict[str, Any]]
) -> List[DataPoint]:
    """
    Process layered knowledge graphs using a configured pipeline.

    Args:
        data_points: List of data points with layered graphs attached
        pipeline_config: List of pipeline step configurations, each with 'type' and other parameters

    Returns:
        The original data points with processed graphs
    """
    # Process each data point
    for data_point in data_points:
        if not hasattr(data_point, "layered_graph"):
            continue

        graph = data_point.layered_graph

        # Process each pipeline step
        for step_config in pipeline_config:
            step_type = step_config.get("type")

            if step_type == "enrich":
                # Enrich the graph
                enrichment_type = step_config.get("enrichment_type", "general")
                content = step_config.get("content")
                parent_layer_ids = step_config.get("parent_layer_ids")

                graph = await enrich_layered_graph(
                    graph, enrichment_type, content, parent_layer_ids
                )

            elif step_type == "store":
                # Store the graph
                adapter = LayeredGraphDBAdapter()
                graph_id = await adapter.store_graph(graph)
                setattr(data_point, "layered_graph_id", graph_id)

            elif step_type == "analyze":
                # Analyze the graph and add metrics
                metrics = {}

                # Get layer metrics
                for layer_id, layer in graph.layers.items():
                    layer_nodes = graph.get_layer_nodes(layer_id)
                    layer_edges = graph.get_layer_edges(layer_id)

                    metrics[str(layer_id)] = {
                        "name": layer.name,
                        "node_count": len(layer_nodes),
                        "edge_count": len(layer_edges),
                        "node_types": {},
                        "edge_types": {},
                    }

                    # Count node types
                    for node in layer_nodes:
                        if node.node_type not in metrics[str(layer_id)]["node_types"]:
                            metrics[str(layer_id)]["node_types"][node.node_type] = 0
                        metrics[str(layer_id)]["node_types"][node.node_type] += 1

                    # Count edge types
                    for edge in layer_edges:
                        if edge.relationship_name not in metrics[str(layer_id)]["edge_types"]:
                            metrics[str(layer_id)]["edge_types"][edge.relationship_name] = 0
                        metrics[str(layer_id)]["edge_types"][edge.relationship_name] += 1

                setattr(data_point, "layered_graph_metrics", metrics)

        # Update the graph on the data point
        data_point.layered_graph = graph

    return data_points
