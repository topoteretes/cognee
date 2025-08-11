"""Ontology adapter implementations."""

import difflib
from typing import List, Tuple, Optional
from collections import deque

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IOntologyAdapter,
    OntologyGraph,
    OntologyNode,
    OntologyEdge,
)

logger = get_logger("OntologyAdapters")


class DefaultOntologyAdapter(IOntologyAdapter):
    """Default implementation of ontology adapter."""

    async def find_matching_nodes(
        self,
        query_text: str,
        ontology: OntologyGraph,
        similarity_threshold: float = 0.8
    ) -> List[OntologyNode]:
        """Find nodes matching query text using simple string similarity."""
        
        matching_nodes = []
        query_lower = query_text.lower()
        
        for node in ontology.nodes:
            # Check name similarity
            name_similarity = self._calculate_similarity(query_lower, node.name.lower())
            
            # Check description similarity
            desc_similarity = 0.0
            if node.description:
                desc_similarity = self._calculate_similarity(query_lower, node.description.lower())
            
            # Check properties similarity
            props_similarity = 0.0
            for prop_value in node.properties.values():
                if isinstance(prop_value, str):
                    prop_sim = self._calculate_similarity(query_lower, prop_value.lower())
                    props_similarity = max(props_similarity, prop_sim)
            
            # Take maximum similarity
            max_similarity = max(name_similarity, desc_similarity, props_similarity)
            
            if max_similarity >= similarity_threshold:
                # Add similarity score to node properties for ranking
                node_copy = OntologyNode(**node.dict())
                node_copy.properties["_similarity_score"] = max_similarity
                matching_nodes.append(node_copy)
        
        # Sort by similarity score
        matching_nodes.sort(key=lambda n: n.properties.get("_similarity_score", 0), reverse=True)
        
        logger.debug(f"Found {len(matching_nodes)} nodes matching '{query_text}'")
        return matching_nodes

    async def get_node_relationships(
        self,
        node_id: str,
        ontology: OntologyGraph,
        max_depth: int = 2
    ) -> List[OntologyEdge]:
        """Get relationships for a specific node."""
        
        relationships = []
        visited = set()
        queue = deque([(node_id, 0)])  # (node_id, depth)
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Find edges where this node is source or target
            for edge in ontology.edges:
                if edge.source_id == current_id:
                    relationships.append(edge)
                    if depth < max_depth:
                        queue.append((edge.target_id, depth + 1))
                
                elif edge.target_id == current_id:
                    relationships.append(edge)
                    if depth < max_depth:
                        queue.append((edge.source_id, depth + 1))
        
        # Remove duplicates
        unique_relationships = []
        seen_edges = set()
        for rel in relationships:
            edge_key = (rel.source_id, rel.target_id, rel.relationship_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_relationships.append(rel)
        
        logger.debug(f"Found {len(unique_relationships)} relationships for node {node_id}")
        return unique_relationships

    async def expand_subgraph(
        self,
        node_ids: List[str],
        ontology: OntologyGraph,
        directed: bool = True
    ) -> Tuple[List[OntologyNode], List[OntologyEdge]]:
        """Expand subgraph around given nodes."""
        
        subgraph_nodes = []
        subgraph_edges = []
        
        # Get all nodes in the node_ids list
        node_map = {node.id: node for node in ontology.nodes}
        included_node_ids = set(node_ids)
        
        # Add initial nodes
        for node_id in node_ids:
            if node_id in node_map:
                subgraph_nodes.append(node_map[node_id])
        
        # Find connected edges and nodes
        for edge in ontology.edges:
            include_edge = False
            
            if directed:
                # Include edge if source is in our set
                if edge.source_id in included_node_ids:
                    include_edge = True
                    # Add target node if not already included
                    if edge.target_id not in included_node_ids and edge.target_id in node_map:
                        subgraph_nodes.append(node_map[edge.target_id])
                        included_node_ids.add(edge.target_id)
            else:
                # Include edge if either source or target is in our set
                if edge.source_id in included_node_ids or edge.target_id in included_node_ids:
                    include_edge = True
                    # Add both nodes if not already included
                    for node_id in [edge.source_id, edge.target_id]:
                        if node_id not in included_node_ids and node_id in node_map:
                            subgraph_nodes.append(node_map[node_id])
                            included_node_ids.add(node_id)
            
            if include_edge:
                subgraph_edges.append(edge)
        
        logger.debug(f"Expanded subgraph with {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges")
        return subgraph_nodes, subgraph_edges

    async def merge_ontologies(
        self,
        ontologies: List[OntologyGraph]
    ) -> OntologyGraph:
        """Merge multiple ontologies."""
        
        if not ontologies:
            raise ValueError("No ontologies to merge")
        
        if len(ontologies) == 1:
            return ontologies[0]
        
        # Create merged ontology
        merged_nodes = []
        merged_edges = []
        merged_metadata = {}
        
        # Keep track of node and edge IDs to avoid duplicates
        seen_node_ids = set()
        seen_edge_ids = set()
        
        # Merge nodes
        for ontology in ontologies:
            for node in ontology.nodes:
                if node.id not in seen_node_ids:
                    merged_nodes.append(node)
                    seen_node_ids.add(node.id)
                else:
                    # Handle duplicate nodes by merging properties
                    existing_node = next(n for n in merged_nodes if n.id == node.id)
                    existing_node.properties.update(node.properties)
        
        # Merge edges
        for ontology in ontologies:
            for edge in ontology.edges:
                edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
                if edge_key not in seen_edge_ids:
                    merged_edges.append(edge)
                    seen_edge_ids.add(edge_key)
        
        # Merge metadata
        for ontology in ontologies:
            merged_metadata.update(ontology.metadata)
        
        merged_metadata["merged_from"] = [ont.id for ont in ontologies]
        from datetime import datetime
        merged_metadata["merge_timestamp"] = datetime.now().isoformat()
        
        merged_ontology = OntologyGraph(
            id=f"merged_{'_'.join([ont.id for ont in ontologies[:3]])}",
            name=f"Merged Ontology ({len(ontologies)} sources)",
            description=f"Merged from: {', '.join([ont.name for ont in ontologies])}",
            format=ontologies[0].format,  # Use format of first ontology
            scope=ontologies[0].scope,    # Use scope of first ontology
            nodes=merged_nodes,
            edges=merged_edges,
            metadata=merged_metadata
        )
        
        logger.info(f"Merged {len(ontologies)} ontologies into one with {len(merged_nodes)} nodes and {len(merged_edges)} edges")
        return merged_ontology

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Use difflib for sequence similarity
        return difflib.SequenceMatcher(None, text1, text2).ratio()


class SemanticOntologyAdapter(DefaultOntologyAdapter):
    """Semantic-aware ontology adapter using embeddings (if available)."""

    def __init__(self):
        super().__init__()
        self.embeddings_available = False
        try:
            # Try to import embedding functionality
            from cognee.infrastructure.llm import get_embedding_engine
            self.get_embedding_engine = get_embedding_engine
            self.embeddings_available = True
        except ImportError:
            logger.warning("Embedding engine not available, falling back to string similarity")

    async def find_matching_nodes(
        self,
        query_text: str,
        ontology: OntologyGraph,
        similarity_threshold: float = 0.8
    ) -> List[OntologyNode]:
        """Find nodes using semantic similarity if embeddings are available."""
        
        if not self.embeddings_available:
            return await super().find_matching_nodes(query_text, ontology, similarity_threshold)
        
        try:
            # Get embedding for query
            embedding_engine = await self.get_embedding_engine()
            query_embedding = await embedding_engine.embed_text(query_text)
            
            matching_nodes = []
            
            for node in ontology.nodes:
                # Create node text for embedding
                node_text = f"{node.name} {node.description or ''}"
                for prop_value in node.properties.values():
                    if isinstance(prop_value, str):
                        node_text += f" {prop_value}"
                
                # Get node embedding
                node_embedding = await embedding_engine.embed_text(node_text)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                
                if similarity >= similarity_threshold:
                    node_copy = OntologyNode(**node.dict())
                    node_copy.properties["_similarity_score"] = similarity
                    matching_nodes.append(node_copy)
            
            # Sort by similarity
            matching_nodes.sort(key=lambda n: n.properties.get("_similarity_score", 0), reverse=True)
            
            logger.debug(f"Found {len(matching_nodes)} nodes using semantic similarity")
            return matching_nodes
            
        except Exception as e:
            logger.warning(f"Semantic similarity failed, falling back to string matching: {e}")
            return await super().find_matching_nodes(query_text, ontology, similarity_threshold)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class GraphOntologyAdapter(DefaultOntologyAdapter):
    """Adapter specialized for graph-based operations."""

    async def get_node_relationships(
        self,
        node_id: str,
        ontology: OntologyGraph,
        max_depth: int = 2
    ) -> List[OntologyEdge]:
        """Enhanced relationship discovery with graph algorithms."""
        
        # Build adjacency lists for faster traversal
        outgoing = {}
        incoming = {}
        
        for edge in ontology.edges:
            if edge.source_id not in outgoing:
                outgoing[edge.source_id] = []
            outgoing[edge.source_id].append(edge)
            
            if edge.target_id not in incoming:
                incoming[edge.target_id] = []
            incoming[edge.target_id].append(edge)
        
        # BFS traversal
        relationships = []
        visited = set()
        queue = deque([(node_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Add outgoing edges
            for edge in outgoing.get(current_id, []):
                relationships.append(edge)
                if depth < max_depth:
                    queue.append((edge.target_id, depth + 1))
            
            # Add incoming edges
            for edge in incoming.get(current_id, []):
                relationships.append(edge)
                if depth < max_depth:
                    queue.append((edge.source_id, depth + 1))
        
        # Remove duplicates and sort by relevance
        unique_relationships = list({edge.id: edge for edge in relationships}.values())
        
        # Sort by edge weight if available, then by relationship type
        unique_relationships.sort(
            key=lambda e: (e.weight or 0, e.relationship_type),
            reverse=True
        )
        
        return unique_relationships

    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        ontology: OntologyGraph
    ) -> List[OntologyEdge]:
        """Find shortest path between two nodes."""
        
        # Build graph
        graph = {}
        for edge in ontology.edges:
            if edge.source_id not in graph:
                graph[edge.source_id] = []
            graph[edge.source_id].append((edge.target_id, edge))
        
        # BFS for shortest path
        queue = deque([(source_id, [])])
        visited = set()
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                return path
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for neighbor_id, edge in graph.get(current_id, []):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [edge]))
        
        return []  # No path found
