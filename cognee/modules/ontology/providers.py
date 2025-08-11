"""Ontology provider implementations."""

import json
import csv
from typing import Dict, Any, Union, Optional
from pathlib import Path

from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.interfaces import (
    IOntologyProvider,
    OntologyGraph,
    OntologyNode,
    OntologyEdge,
    OntologyFormat,
    OntologyScope,
    OntologyContext,
)

logger = get_logger("OntologyProviders")


class RDFOntologyProvider(IOntologyProvider):
    """Provider for RDF/OWL ontologies."""

    def __init__(self):
        try:
            from rdflib import Graph, URIRef, RDF, RDFS, OWL
            self.Graph = Graph
            self.URIRef = URIRef
            self.RDF = RDF
            self.RDFS = RDFS
            self.OWL = OWL
            self.available = True
        except ImportError:
            logger.warning("rdflib not available, RDF support disabled")
            self.available = False

    async def load_ontology(
        self, 
        source: Union[str, Dict[str, Any]], 
        context: Optional[OntologyContext] = None
    ) -> OntologyGraph:
        """Load ontology from RDF/OWL file."""
        
        if not self.available:
            raise ImportError("rdflib is required for RDF ontology support")
        
        if isinstance(source, dict):
            file_path = source.get("file_path")
        else:
            file_path = source
        
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(f"RDF file not found: {file_path}")
        
        # Parse RDF graph
        rdf_graph = self.Graph()
        rdf_graph.parse(file_path)
        
        # Convert to our ontology format
        nodes = []
        edges = []
        
        # Extract classes and individuals
        for cls in rdf_graph.subjects(self.RDF.type, self.OWL.Class):
            node = OntologyNode(
                id=self._uri_to_id(cls),
                name=self._extract_name(cls),
                type="class",
                category="owl_class",
                properties=self._extract_node_properties(cls, rdf_graph)
            )
            nodes.append(node)
        
        # Extract individuals
        for individual in rdf_graph.subjects(self.RDF.type, None):
            if not any(rdf_graph.triples((individual, self.RDF.type, self.OWL.Class))):
                node = OntologyNode(
                    id=self._uri_to_id(individual),
                    name=self._extract_name(individual),
                    type="individual",
                    category="owl_individual",
                    properties=self._extract_node_properties(individual, rdf_graph)
                )
                nodes.append(node)
        
        # Extract relationships
        for s, p, o in rdf_graph:
            if p != self.RDF.type:  # Skip type relationships
                edge = OntologyEdge(
                    id=f"{self._uri_to_id(s)}_{self._uri_to_id(p)}_{self._uri_to_id(o)}",
                    source_id=self._uri_to_id(s),
                    target_id=self._uri_to_id(o),
                    relationship_type=self._extract_name(p),
                    properties={"predicate_uri": str(p)}
                )
                edges.append(edge)
        
        ontology = OntologyGraph(
            id=f"rdf_{Path(file_path).stem}",
            name=Path(file_path).stem,
            description=f"RDF ontology loaded from {file_path}",
            format=OntologyFormat.RDF_XML,
            scope=OntologyScope.DOMAIN,
            nodes=nodes,
            edges=edges,
            metadata={"source_file": file_path, "triple_count": len(rdf_graph)}
        )
        
        logger.info(f"Loaded RDF ontology with {len(nodes)} nodes and {len(edges)} edges")
        return ontology

    async def save_ontology(
        self, 
        ontology: OntologyGraph, 
        destination: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Save ontology to RDF/OWL file."""
        
        if not self.available:
            raise ImportError("rdflib is required for RDF ontology support")
        
        # Convert back to RDF
        rdf_graph = self.Graph()
        
        # Add nodes
        for node in ontology.nodes:
            uri = self.URIRef(f"http://example.org/ontology#{node.id}")
            if node.type == "class":
                rdf_graph.add((uri, self.RDF.type, self.OWL.Class))
            else:
                # Add as individual of some class
                class_uri = self.URIRef(f"http://example.org/ontology#{node.type}")
                rdf_graph.add((uri, self.RDF.type, class_uri))
        
        # Add edges
        for edge in ontology.edges:
            s_uri = self.URIRef(f"http://example.org/ontology#{edge.source_id}")
            p_uri = self.URIRef(f"http://example.org/ontology#{edge.relationship_type}")
            o_uri = self.URIRef(f"http://example.org/ontology#{edge.target_id}")
            rdf_graph.add((s_uri, p_uri, o_uri))
        
        # Serialize to file
        try:
            rdf_graph.serialize(destination=destination, format='xml')
            logger.info(f"Saved RDF ontology to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to save RDF ontology: {e}")
            return False

    def supports_format(self, format: OntologyFormat) -> bool:
        """Check if provider supports given format."""
        return self.available and format in [OntologyFormat.RDF_XML, OntologyFormat.OWL]

    async def validate_ontology(self, ontology: OntologyGraph) -> bool:
        """Validate RDF ontology structure."""
        # Basic validation - could be enhanced with OWL reasoning
        node_ids = {node.id for node in ontology.nodes}
        
        for edge in ontology.edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                return False
        
        return True

    def _uri_to_id(self, uri) -> str:
        """Convert URI to simple ID."""
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.rstrip("/").split("/")[-1]

    def _extract_name(self, uri) -> str:
        """Extract readable name from URI."""
        return self._uri_to_id(uri).replace("_", " ").title()

    def _extract_node_properties(self, uri, graph) -> Dict[str, Any]:
        """Extract additional properties for a node."""
        props = {}
        
        # Get labels
        for label in graph.objects(uri, self.RDFS.label):
            props["label"] = str(label)
        
        # Get comments
        for comment in graph.objects(uri, self.RDFS.comment):
            props["comment"] = str(comment)
        
        return props


class JSONOntologyProvider(IOntologyProvider):
    """Provider for JSON-based ontologies."""

    async def load_ontology(
        self, 
        source: Union[str, Dict[str, Any]], 
        context: Optional[OntologyContext] = None
    ) -> OntologyGraph:
        """Load ontology from JSON file or dict."""
        
        if isinstance(source, str):
            # Load from file
            with open(source, 'r') as f:
                data = json.load(f)
            ontology_id = f"json_{Path(source).stem}"
            source_file = source
        else:
            # Use provided dict
            data = source
            ontology_id = data.get("id", "json_ontology")
            source_file = None
        
        # Parse nodes
        nodes = []
        for node_data in data.get("nodes", []):
            node = OntologyNode(
                id=node_data["id"],
                name=node_data.get("name", node_data["id"]),
                type=node_data.get("type", "entity"),
                description=node_data.get("description", ""),
                category=node_data.get("category", "general"),
                properties=node_data.get("properties", {})
            )
            nodes.append(node)
        
        # Parse edges
        edges = []
        for edge_data in data.get("edges", []):
            edge = OntologyEdge(
                id=edge_data.get("id", f"{edge_data['source']}_{edge_data['target']}"),
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                relationship_type=edge_data.get("relationship", "related_to"),
                properties=edge_data.get("properties", {}),
                weight=edge_data.get("weight")
            )
            edges.append(edge)
        
        ontology = OntologyGraph(
            id=ontology_id,
            name=data.get("name", ontology_id),
            description=data.get("description", "JSON-based ontology"),
            format=OntologyFormat.JSON,
            scope=OntologyScope.DOMAIN,
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {"source_file": source_file})
        )
        
        logger.info(f"Loaded JSON ontology with {len(nodes)} nodes and {len(edges)} edges")
        return ontology

    async def save_ontology(
        self, 
        ontology: OntologyGraph, 
        destination: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Save ontology to JSON file."""
        
        data = {
            "id": ontology.id,
            "name": ontology.name,
            "description": ontology.description,
            "format": ontology.format.value,
            "scope": ontology.scope.value,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "description": node.description,
                    "category": node.category,
                    "properties": node.properties
                }
                for node in ontology.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relationship": edge.relationship_type,
                    "properties": edge.properties,
                    "weight": edge.weight
                }
                for edge in ontology.edges
            ],
            "metadata": ontology.metadata
        }
        
        try:
            with open(destination, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved JSON ontology to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON ontology: {e}")
            return False

    def supports_format(self, format: OntologyFormat) -> bool:
        """Check if provider supports given format."""
        return format == OntologyFormat.JSON

    async def validate_ontology(self, ontology: OntologyGraph) -> bool:
        """Validate JSON ontology structure."""
        node_ids = {node.id for node in ontology.nodes}
        
        for edge in ontology.edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                return False
        
        return True


class CSVOntologyProvider(IOntologyProvider):
    """Provider for CSV-based ontologies."""

    async def load_ontology(
        self, 
        source: Union[str, Dict[str, Any]], 
        context: Optional[OntologyContext] = None
    ) -> OntologyGraph:
        """Load ontology from CSV files."""
        
        if isinstance(source, dict):
            nodes_file = source.get("nodes_file")
            edges_file = source.get("edges_file")
        else:
            # Assume single file or directory
            source_path = Path(source)
            if source_path.is_dir():
                nodes_file = source_path / "nodes.csv"
                edges_file = source_path / "edges.csv"
            else:
                nodes_file = source
                edges_file = None
        
        # Load nodes
        nodes = []
        if nodes_file and Path(nodes_file).exists():
            with open(nodes_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = OntologyNode(
                        id=row["id"],
                        name=row.get("name", row["id"]),
                        type=row.get("type", "entity"),
                        description=row.get("description", ""),
                        category=row.get("category", "general"),
                        properties={k: v for k, v in row.items() 
                                  if k not in ["id", "name", "type", "description", "category"]}
                    )
                    nodes.append(node)
        
        # Load edges
        edges = []
        if edges_file and Path(edges_file).exists():
            with open(edges_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edge = OntologyEdge(
                        id=row.get("id", f"{row['source']}_{row['target']}"),
                        source_id=row["source"],
                        target_id=row["target"],
                        relationship_type=row.get("relationship", "related_to"),
                        properties={k: v for k, v in row.items() 
                                  if k not in ["id", "source", "target", "relationship"]},
                        weight=float(row["weight"]) if row.get("weight") else None
                    )
                    edges.append(edge)
        
        ontology = OntologyGraph(
            id=f"csv_{Path(nodes_file).stem}" if nodes_file else "csv_ontology",
            name=f"CSV Ontology",
            description="CSV-based ontology",
            format=OntologyFormat.CSV,
            scope=OntologyScope.DOMAIN,
            nodes=nodes,
            edges=edges,
            metadata={"nodes_file": str(nodes_file), "edges_file": str(edges_file)}
        )
        
        logger.info(f"Loaded CSV ontology with {len(nodes)} nodes and {len(edges)} edges")
        return ontology

    async def save_ontology(
        self, 
        ontology: OntologyGraph, 
        destination: str,
        context: Optional[OntologyContext] = None
    ) -> bool:
        """Save ontology to CSV files."""
        
        dest_path = Path(destination)
        if dest_path.suffix == ".csv":
            # Single file - save nodes only
            nodes_file = destination
            edges_file = None
        else:
            # Directory - save separate files
            dest_path.mkdir(exist_ok=True)
            nodes_file = dest_path / "nodes.csv"
            edges_file = dest_path / "edges.csv"
        
        try:
            # Save nodes
            if ontology.nodes:
                all_properties = set()
                for node in ontology.nodes:
                    all_properties.update(node.properties.keys())
                
                fieldnames = ["id", "name", "type", "description", "category"] + list(all_properties)
                
                with open(nodes_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for node in ontology.nodes:
                        row = {
                            "id": node.id,
                            "name": node.name,
                            "type": node.type,
                            "description": node.description,
                            "category": node.category,
                            **node.properties
                        }
                        writer.writerow(row)
            
            # Save edges
            if edges_file and ontology.edges:
                all_properties = set()
                for edge in ontology.edges:
                    all_properties.update(edge.properties.keys())
                
                fieldnames = ["id", "source", "target", "relationship", "weight"] + list(all_properties)
                
                with open(edges_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for edge in ontology.edges:
                        row = {
                            "id": edge.id,
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "relationship": edge.relationship_type,
                            "weight": edge.weight,
                            **edge.properties
                        }
                        writer.writerow(row)
            
            logger.info(f"Saved CSV ontology to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CSV ontology: {e}")
            return False

    def supports_format(self, format: OntologyFormat) -> bool:
        """Check if provider supports given format."""
        return format == OntologyFormat.CSV

    async def validate_ontology(self, ontology: OntologyGraph) -> bool:
        """Validate CSV ontology structure."""
        node_ids = {node.id for node in ontology.nodes}
        
        for edge in ontology.edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                return False
        
        return True
