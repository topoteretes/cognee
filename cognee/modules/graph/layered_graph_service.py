"""
Layered Knowledge Graph Service for Cognee.

This module provides services for working with layered knowledge graphs.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Union, Tuple


from cognee.shared.data_models import (
    LayeredKnowledgeGraph,
    KnowledgeGraph,
    Layer,
    Node,
    Edge
)
from cognee.modules.graph.layered_graph_builder import LayeredGraphBuilder

logger = logging.getLogger(__name__)


class LayeredGraphService:
    """
    Service for working with layered knowledge graphs in Cognee.
    
    This service provides methods for manipulating, merging, and querying 
    layered knowledge graphs.
    """
    
    @staticmethod
    async def merge_layers(
        layered_graph: LayeredKnowledgeGraph,
        layer_ids: List[str],
        new_layer_name: str,
        new_layer_description: str,
        new_layer_type: str = "merged",
        handle_conflicts: str = "overwrite"
    ) -> Tuple[str, LayeredKnowledgeGraph]:
        """
        Merge multiple layers into a new layer in the graph.
        
        Args:
            layered_graph: The layered knowledge graph
            layer_ids: List of layer IDs to merge
            new_layer_name: Name for the merged layer
            new_layer_description: Description for the merged layer
            new_layer_type: Type for the merged layer
            handle_conflicts: How to handle node/edge conflicts: "overwrite", "keep_first", or "keep_original"
            
        Returns:
            Tuple of (new layer ID, updated layered graph)
        """
        # Create a copy of the layered graph to avoid modifying the original
        builder = LayeredGraphBuilder(name=layered_graph.name, description=layered_graph.description)
        
        # Copy existing layers
        for layer in layered_graph.layers:
            if layer.id not in layer_ids:  # Skip layers that will be merged
                parent_layers = [p for p in layer.parent_layers if p not in layer_ids]
                builder.create_layer(
                    name=layer.name,
                    description=layer.description,
                    layer_type=layer.layer_type,
                    parent_layers=parent_layers,
                    layer_id=layer.id,
                    properties=layer.properties
                )
        
        # Create the new merged layer
        new_layer_id = builder.create_layer(
            name=new_layer_name,
            description=new_layer_description,
            layer_type=new_layer_type
        )
        
        # Keep track of processed nodes and edges to handle conflicts
        processed_nodes: Set[str] = set()
        processed_edges: Set[tuple] = set()
        
        # Merge nodes and edges from specified layers
        for layer_id in layer_ids:
            layer_graph = layered_graph.get_layer_graph(layer_id)
            
            # Process nodes
            for node in layer_graph.nodes:
                if node.id in processed_nodes and handle_conflicts == "keep_first":
                    continue
                    
                builder.add_node_to_layer(
                    layer_id=new_layer_id,
                    node_id=node.id,
                    name=node.name,
                    node_type=node.type,
                    description=node.description,
                    properties=node.properties
                )
                processed_nodes.add(node.id)
            
            # Process edges
            for edge in layer_graph.edges:
                edge_key = (edge.source_node_id, edge.target_node_id, edge.relationship_name)
                
                if edge_key in processed_edges and handle_conflicts == "keep_first":
                    continue
                    
                builder.add_edge_to_layer(
                    layer_id=new_layer_id,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    relationship_name=edge.relationship_name,
                    properties=edge.properties
                )
                processed_edges.add(edge_key)
        
        return new_layer_id, builder.build()
    
    @staticmethod
    async def diff_layers(
        layered_graph: LayeredKnowledgeGraph,
        base_layer_id: str,
        comparison_layer_id: str
    ) -> Dict[str, Any]:
        """
        Compare two layers and return their differences.
        
        Args:
            layered_graph: The layered knowledge graph
            base_layer_id: ID of the base layer for comparison
            comparison_layer_id: ID of the layer to compare against the base
            
        Returns:
            Dictionary containing differences between the layers
        """
        base_graph = layered_graph.get_layer_graph(base_layer_id)
        comparison_graph = layered_graph.get_layer_graph(comparison_layer_id)
        
        # Get node IDs and edge keys for easy comparison
        base_node_ids = {node.id for node in base_graph.nodes}
        comparison_node_ids = {node.id for node in comparison_graph.nodes}
        
        base_edge_keys = {
            (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            for edge in base_graph.edges
        }
        comparison_edge_keys = {
            (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            for edge in comparison_graph.edges
        }
        
        # Calculate differences
        added_nodes = comparison_node_ids - base_node_ids
        removed_nodes = base_node_ids - comparison_node_ids
        common_nodes = base_node_ids.intersection(comparison_node_ids)
        
        added_edges = comparison_edge_keys - base_edge_keys
        removed_edges = base_edge_keys - comparison_edge_keys
        common_edges = base_edge_keys.intersection(comparison_edge_keys)
        
        # Find modified nodes (same ID but different properties)
        modified_nodes = []
        for node_id in common_nodes:
            base_node = next((node for node in base_graph.nodes if node.id == node_id), None)
            comparison_node = next((node for node in comparison_graph.nodes if node.id == node_id), None)
            
            if base_node is not None and comparison_node is not None:
                # Check if properties differ
                if (base_node.name != comparison_node.name or
                    base_node.type != comparison_node.type or
                    base_node.description != comparison_node.description or
                    base_node.properties != comparison_node.properties):
                    modified_nodes.append(node_id)
        
        # Find modified edges (same key but different properties)
        modified_edges = []
        for edge_key in common_edges:
            base_edge = next((edge for edge in base_graph.edges 
                              if (edge.source_node_id, edge.target_node_id, edge.relationship_name) == edge_key), None)
            comparison_edge = next((edge for edge in comparison_graph.edges 
                                   if (edge.source_node_id, edge.target_node_id, edge.relationship_name) == edge_key), None)
            
            if base_edge is not None and comparison_edge is not None:
                # Check if properties differ
                if base_edge.properties != comparison_edge.properties:
                    modified_edges.append(edge_key)
        
        return {
            "added_nodes": list(added_nodes),
            "removed_nodes": list(removed_nodes),
            "modified_nodes": modified_nodes,
            "common_nodes": list(common_nodes),
            "added_edges": list(added_edges),
            "removed_edges": list(removed_edges),
            "modified_edges": modified_edges,
            "common_edges": list(common_edges),
            "node_count_diff": len(comparison_graph.nodes) - len(base_graph.nodes),
            "edge_count_diff": len(comparison_graph.edges) - len(base_graph.edges)
        }
    
    @staticmethod
    async def extract_subgraph(
        layered_graph: LayeredKnowledgeGraph,
        layer_ids: List[str] = None,
        include_cumulative: bool = False,
        node_filter: Optional[callable] = None,
        edge_filter: Optional[callable] = None
    ) -> KnowledgeGraph:
        """
        Extract a subgraph from a layered graph based on specified filters.
        
        Args:
            layered_graph: The layered knowledge graph
            layer_ids: List of layer IDs to include (None = all layers)
            include_cumulative: Whether to include parent layers in extraction
            node_filter: Optional function to filter nodes (takes Node, returns bool)
            edge_filter: Optional function to filter edges (takes Edge, returns bool)
            
        Returns:
            Knowledge graph containing the filtered subgraph
        """
        # If no layer IDs specified, use all layers
        if layer_ids is None:
            layer_ids = [layer.id for layer in layered_graph.layers]
            
        # Initialize lists for nodes and edges
        nodes = []
        edges = []
        
        # Process each specified layer
        for layer_id in layer_ids:
            # Get graph for this layer (cumulative or not)
            if include_cumulative:
                layer_graph = layered_graph.get_cumulative_layer_graph(layer_id)
            else:
                layer_graph = layered_graph.get_layer_graph(layer_id)
            
            # Apply node filter if provided
            if node_filter is not None:
                filtered_nodes = [node for node in layer_graph.nodes if node_filter(node)]
            else:
                filtered_nodes = layer_graph.nodes
                
            # Get IDs of nodes that passed the filter
            filtered_node_ids = {node.id for node in filtered_nodes}
            
            # Add filtered nodes to result
            nodes.extend(filtered_nodes)
            
            # Apply edge filter if provided, and ensure edges connect to filtered nodes
            if edge_filter is not None:
                filtered_edges = [
                    edge for edge in layer_graph.edges 
                    if edge_filter(edge) and 
                    edge.source_node_id in filtered_node_ids and 
                    edge.target_node_id in filtered_node_ids
                ]
            else:
                filtered_edges = [
                    edge for edge in layer_graph.edges 
                    if edge.source_node_id in filtered_node_ids and 
                    edge.target_node_id in filtered_node_ids
                ]
                
            # Add filtered edges to result
            edges.extend(filtered_edges)
        
        # Remove duplicate nodes and edges based on IDs/keys
        unique_nodes = {}
        for node in nodes:
            unique_nodes[node.id] = node
            
        unique_edges = {}
        for edge in edges:
            edge_key = (edge.source_node_id, edge.target_node_id, edge.relationship_name)
            unique_edges[edge_key] = edge
        
        # Create and return the knowledge graph
        return KnowledgeGraph(
            nodes=list(unique_nodes.values()),
            edges=list(unique_edges.values()),
            name=f"Subgraph from {layered_graph.name}",
            description=f"Subgraph extracted from {layered_graph.name} with {len(layer_ids)} layer(s)"
        )
    
    @staticmethod
    async def analyze_layer_dependencies(layered_graph: LayeredKnowledgeGraph) -> Dict[str, Any]:
        """
        Analyze dependencies between layers and return a dependency structure.
        
        Args:
            layered_graph: The layered knowledge graph
            
        Returns:
            Dictionary containing layer dependency analysis
        """
        # Initialize dictionaries for dependency analysis
        dependencies = {}  # layer_id -> set of direct parent layer IDs
        reverse_dependencies = {}  # layer_id -> set of direct child layer IDs
        all_dependencies = {}  # layer_id -> set of all ancestor layer IDs
        layer_depth = {}  # layer_id -> depth in dependency hierarchy
        
        # Create lookup for layers by ID
        layers_by_id = {layer.id: layer for layer in layered_graph.layers}
        
        # Build direct dependencies and reverse dependencies
        for layer in layered_graph.layers:
            dependencies[layer.id] = set(layer.parent_layers)
            
            # Initialize reverse dependencies sets
            for parent_id in layer.parent_layers:
                if parent_id not in reverse_dependencies:
                    reverse_dependencies[parent_id] = set()
                reverse_dependencies[parent_id].add(layer.id)
        
        # Initialize reverse dependencies for layers with no children
        for layer_id in dependencies:
            if layer_id not in reverse_dependencies:
                reverse_dependencies[layer_id] = set()
        
        # Calculate all dependencies using depth-first search
        def get_all_dependencies(layer_id):
            if layer_id in all_dependencies:
                return all_dependencies[layer_id]
                
            all_deps = set()
            for parent_id in dependencies.get(layer_id, set()):
                all_deps.add(parent_id)
                all_deps.update(get_all_dependencies(parent_id))
                
            all_dependencies[layer_id] = all_deps
            return all_deps
        
        # Calculate all dependencies for each layer
        for layer_id in dependencies:
            get_all_dependencies(layer_id)
        
        # Calculate layer depths
        roots = [layer_id for layer_id, parents in dependencies.items() if not parents]
        
        # Set depth 0 for root layers
        for root in roots:
            layer_depth[root] = 0
            
        # Calculate depths using breadth-first search
        visited = set(roots)
        queue = [(root, 0) for root in roots]
        
        while queue:
            layer_id, depth = queue.pop(0)
            
            for child_id in reverse_dependencies.get(layer_id, set()):
                # Calculate child depth as max(current depth + 1, existing depth)
                child_depth = depth + 1
                if child_id in layer_depth:
                    child_depth = max(child_depth, layer_depth[child_id])
                    
                layer_depth[child_id] = child_depth
                
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, child_depth))
        
        # Group layers by depth
        layers_by_depth = {}
        for layer_id, depth in layer_depth.items():
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer_id)
            
        # Check for cycles (if a layer is not visited, it's part of a cycle)
        cycles = [layer_id for layer_id in dependencies if layer_id not in visited]
        
        return {
            "root_layers": roots,
            "leaf_layers": [layer_id for layer_id, children in reverse_dependencies.items() if not children],
            "dependencies": dependencies,
            "reverse_dependencies": reverse_dependencies,
            "all_dependencies": all_dependencies,
            "layer_depth": layer_depth,
            "layers_by_depth": layers_by_depth,
            "max_depth": max(layer_depth.values()) if layer_depth else 0,
            "has_cycles": len(cycles) > 0,
            "cycle_layers": cycles
        }
    
    @staticmethod
    async def filter_graph_by_relationship_types(
        layered_graph: LayeredKnowledgeGraph,
        relationship_types: List[str],
        include_only: bool = True,
        layer_ids: List[str] = None
    ) -> KnowledgeGraph:
        """
        Filter a layered graph to include or exclude specific relationship types.
        
        Args:
            layered_graph: The layered knowledge graph
            relationship_types: List of relationship types to filter by
            include_only: If True, include only these relationships; if False, exclude them
            layer_ids: Optional list of layer IDs to filter (None = all layers)
            
        Returns:
            Filtered knowledge graph
        """
        # Define edge filter based on relationship types
        def edge_filter(edge):
            relationship_match = edge.relationship_name in relationship_types
            return relationship_match if include_only else not relationship_match
            
        # Use extract_subgraph with the edge filter
        return await LayeredGraphService.extract_subgraph(
            layered_graph=layered_graph,
            layer_ids=layer_ids,
            include_cumulative=True,
            edge_filter=edge_filter
        )
    
    @staticmethod
    async def sort_layers_topologically(layered_graph: LayeredKnowledgeGraph) -> List[str]:
        """
        Sort layers in topological order (parents before children).
        
        Args:
            layered_graph: The layered knowledge graph
            
        Returns:
            List of layer IDs in topological order
        """
        # First, get the dependency analysis
        analysis = await LayeredGraphService.analyze_layer_dependencies(layered_graph)
        
        # Check for cycles
        if analysis["has_cycles"]:
            logger.warning("Graph has cycles, topological sort may not be complete")
            
        # Use the layers_by_depth to create a topological order
        layers_by_depth = analysis["layers_by_depth"]
        sorted_depths = sorted(layers_by_depth.keys())
        
        # Flatten the layers by depth
        topological_order = []
        for depth in sorted_depths:
            topological_order.extend(layers_by_depth[depth])
            
        return topological_order
    
    @staticmethod
    async def find_nodes_by_property(
        layered_graph: LayeredKnowledgeGraph,
        property_name: str,
        property_value: Any,
        layer_ids: List[str] = None,
        include_cumulative: bool = False
    ) -> List[Node]:
        """
        Find nodes that have a specific property value.
        
        Args:
            layered_graph: The layered knowledge graph
            property_name: Name of the property to search
            property_value: Value of the property to match
            layer_ids: Optional list of layer IDs to search (None = all layers)
            include_cumulative: Whether to include parent layers in the search
            
        Returns:
            List of nodes matching the property value
        """
        # Define node filter for the property
        def node_filter(node):
            if property_name == "id":
                return node.id == property_value
            elif property_name == "name":
                return node.name == property_value
            elif property_name == "type":
                return node.type == property_value
            elif property_name == "description":
                return property_value in node.description
            else:
                # Check node properties
                return (property_name in node.properties and 
                        node.properties[property_name] == property_value)
                
        # Use extract_subgraph with the node filter
        subgraph = await LayeredGraphService.extract_subgraph(
            layered_graph=layered_graph,
            layer_ids=layer_ids,
            include_cumulative=include_cumulative,
            node_filter=node_filter
        )
        
        return subgraph.nodes
    
    @staticmethod
    async def calculate_layer_metrics(layered_graph: LayeredKnowledgeGraph) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for each layer in the graph.
        
        Args:
            layered_graph: The layered knowledge graph
            
        Returns:
            Dictionary mapping layer IDs to dictionaries of metrics
        """
        metrics = {}
        
        # Get dependency analysis
        analysis = await LayeredGraphService.analyze_layer_dependencies(layered_graph)
        
        # Calculate metrics for each layer
        for layer in layered_graph.layers:
            layer_id = layer.id
            layer_graph = layered_graph.get_layer_graph(layer_id)
            cumulative_graph = layered_graph.get_cumulative_layer_graph(layer_id)
            
            # Count node types in this layer
            node_types = {}
            for node in layer_graph.nodes:
                if node.type not in node_types:
                    node_types[node.type] = 0
                node_types[node.type] += 1
                
            # Count relationship types in this layer
            relationship_types = {}
            for edge in layer_graph.edges:
                if edge.relationship_name not in relationship_types:
                    relationship_types[edge.relationship_name] = 0
                relationship_types[edge.relationship_name] += 1
                
            # Calculate density (ratio of actual to possible edges)
            node_count = len(layer_graph.nodes)
            edge_count = len(layer_graph.edges)
            possible_edges = node_count * (node_count - 1) if node_count > 1 else 0
            density = edge_count / possible_edges if possible_edges > 0 else 0
            
            # Store metrics for this layer
            metrics[layer_id] = {
                "node_count": node_count,
                "edge_count": edge_count,
                "node_types": node_types,
                "relationship_types": relationship_types,
                "density": density,
                "parent_count": len(layer.parent_layers),
                "child_count": len(analysis["reverse_dependencies"].get(layer_id, set())),
                "depth": analysis["layer_depth"].get(layer_id, 0),
                "is_root": len(layer.parent_layers) == 0,
                "is_leaf": len(analysis["reverse_dependencies"].get(layer_id, set())) == 0,
                "cumulative_node_count": len(cumulative_graph.nodes),
                "cumulative_edge_count": len(cumulative_graph.edges),
                "contribution_node_count": len(layer_graph.nodes),
                "contribution_edge_count": len(layer_graph.edges)
            }
            
            # Calculate the contribution ratio (layer's nodes/edges as percentage of cumulative)
            if metrics[layer_id]["cumulative_node_count"] > 0:
                metrics[layer_id]["node_contribution_ratio"] = (
                    metrics[layer_id]["contribution_node_count"] / 
                    metrics[layer_id]["cumulative_node_count"]
                )
            else:
                metrics[layer_id]["node_contribution_ratio"] = 0
                
            if metrics[layer_id]["cumulative_edge_count"] > 0:
                metrics[layer_id]["edge_contribution_ratio"] = (
                    metrics[layer_id]["contribution_edge_count"] / 
                    metrics[layer_id]["cumulative_edge_count"]
                )
            else:
                metrics[layer_id]["edge_contribution_ratio"] = 0
        
        return metrics 