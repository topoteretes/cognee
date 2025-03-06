"""
Advanced layered knowledge graph example demonstrating integration with other Cognee components.

This script shows how to build, analyze, and utilize layered knowledge graphs 
in a more complex scenario using various Cognee components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import datetime

from cognee.shared.data_models import (
    KnowledgeGraph,
    LayeredKnowledgeGraph,
    Layer,
    Node,
    Edge
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedLayeredGraphDemo:
    """
    Advanced demonstration of layered graph capabilities.
    """
    
    def __init__(self):
        """Initialize the demo with an empty layered graph."""
        self.layered_graph = None
        
    async def build_layered_knowledge_graph(self) -> LayeredKnowledgeGraph:
        """
        Build a multi-layer knowledge graph about a software system architecture.
        
        Returns:
            A layered knowledge graph with multiple interconnected layers
        """
        # Create the base KnowledgeGraph
        base_graph = KnowledgeGraph(nodes=[], edges=[], name="System Architecture", description="A layered architecture of a software system")
        
        # Initialize the layered graph
        layered_graph = LayeredKnowledgeGraph(
            base_graph=base_graph,
            layers=[],
            name="Software System Architecture",
            description="A layered representation of software components and their interactions"
        )
        
        # Layer 1: Infrastructure Layer
        infra_layer = Layer(
            id="infrastructure",
            name="Infrastructure Layer",
            description="Contains servers, databases, and network components",
            layer_type="base",
            parent_layers=[]
        )
        layered_graph.add_layer(infra_layer)
        
        # Add infrastructure nodes
        nodes = [
            Node(id="server1", name="Application Server", type="Server", description="Main application server"),
            Node(id="db1", name="Database Server", type="Database", description="PostgreSQL database server"),
            Node(id="storage1", name="Storage Server", type="Storage", description="S3-compatible storage"),
            Node(id="network1", name="Network Switch", type="Network", description="Main network switch")
        ]
        
        for node in nodes:
            layered_graph.add_node_to_layer(node, "infrastructure")
        
        # Add infrastructure edges
        edges = [
            Edge(source_node_id="server1", target_node_id="db1", relationship_name="CONNECTS_TO"),
            Edge(source_node_id="server1", target_node_id="storage1", relationship_name="CONNECTS_TO"),
            Edge(source_node_id="db1", target_node_id="network1", relationship_name="CONNECTS_TO"),
            Edge(source_node_id="storage1", target_node_id="network1", relationship_name="CONNECTS_TO")
        ]
        
        for edge in edges:
            layered_graph.add_edge_to_layer(edge, "infrastructure")
        
        # Layer 2: Software Layer
        software_layer = Layer(
            id="software",
            name="Software Layer",
            description="Software components and applications",
            layer_type="application",
            parent_layers=["infrastructure"]
        )
        layered_graph.add_layer(software_layer)
        
        # Add software nodes
        nodes = [
            Node(id="webapp", name="Web Application", type="Application", description="Main web application"),
            Node(id="api", name="API Service", type="Service", description="RESTful API service"),
            Node(id="workers", name="Worker Processes", type="Service", description="Background workers"),
            Node(id="cache", name="Caching Service", type="Service", description="Redis caching service")
        ]
        
        for node in nodes:
            layered_graph.add_node_to_layer(node, "software")
        
        # Add software edges
        edges = [
            Edge(source_node_id="webapp", target_node_id="api", relationship_name="DEPENDS_ON"),
            Edge(source_node_id="webapp", target_node_id="cache", relationship_name="USES"),
            Edge(source_node_id="api", target_node_id="workers", relationship_name="TRIGGERS"),
            Edge(source_node_id="api", target_node_id="server1", relationship_name="HOSTED_ON"),
            Edge(source_node_id="workers", target_node_id="server1", relationship_name="HOSTED_ON"),
            Edge(source_node_id="cache", target_node_id="server1", relationship_name="HOSTED_ON"),
            Edge(source_node_id="api", target_node_id="db1", relationship_name="ACCESSES")
        ]
        
        for edge in edges:
            layered_graph.add_edge_to_layer(edge, "software")
        
        # Layer 3: Business Logic Layer
        business_layer = Layer(
            id="business",
            name="Business Logic Layer",
            description="Business entities and processes",
            layer_type="domain",
            parent_layers=["software"]
        )
        layered_graph.add_layer(business_layer)
        
        # Add business nodes
        nodes = [
            Node(id="user_mgmt", name="User Management", type="Module", description="User management module"),
            Node(id="order_proc", name="Order Processing", type="Module", description="Order processing module"),
            Node(id="payment", name="Payment Processing", type="Module", description="Payment processing module"),
            Node(id="reporting", name="Reporting", type="Module", description="Reporting and analytics module")
        ]
        
        for node in nodes:
            layered_graph.add_node_to_layer(node, "business")
        
        # Add business edges
        edges = [
            Edge(source_node_id="user_mgmt", target_node_id="webapp", relationship_name="PART_OF"),
            Edge(source_node_id="order_proc", target_node_id="api", relationship_name="EXPOSED_BY"),
            Edge(source_node_id="payment", target_node_id="api", relationship_name="EXPOSED_BY"),
            Edge(source_node_id="reporting", target_node_id="workers", relationship_name="EXECUTED_BY"),
            Edge(source_node_id="order_proc", target_node_id="payment", relationship_name="USES"),
            Edge(source_node_id="reporting", target_node_id="order_proc", relationship_name="ANALYZES")
        ]
        
        for edge in edges:
            layered_graph.add_edge_to_layer(edge, "business")
        
        # Layer 4: User Interface Layer
        ui_layer = Layer(
            id="ui",
            name="User Interface Layer",
            description="UI components and interactions",
            layer_type="presentation",
            parent_layers=["business"]
        )
        layered_graph.add_layer(ui_layer)
        
        # Add UI nodes
        nodes = [
            Node(id="login_page", name="Login Page", type="UI Component", description="User login interface"),
            Node(id="dashboard", name="Dashboard", type="UI Component", description="Main user dashboard"),
            Node(id="order_form", name="Order Form", type="UI Component", description="Order creation form"),
            Node(id="reports_view", name="Reports View", type="UI Component", description="Reports and analytics view")
        ]
        
        for node in nodes:
            layered_graph.add_node_to_layer(node, "ui")
        
        # Add UI edges
        edges = [
            Edge(source_node_id="login_page", target_node_id="user_mgmt", relationship_name="INTERACTS_WITH"),
            Edge(source_node_id="dashboard", target_node_id="reporting", relationship_name="DISPLAYS"),
            Edge(source_node_id="order_form", target_node_id="order_proc", relationship_name="SUBMITS_TO"),
            Edge(source_node_id="reports_view", target_node_id="reporting", relationship_name="VISUALIZES"),
            Edge(source_node_id="login_page", target_node_id="webapp", relationship_name="PART_OF"),
            Edge(source_node_id="dashboard", target_node_id="webapp", relationship_name="PART_OF"),
            Edge(source_node_id="order_form", target_node_id="webapp", relationship_name="PART_OF"),
            Edge(source_node_id="reports_view", target_node_id="webapp", relationship_name="PART_OF")
        ]
        
        for edge in edges:
            layered_graph.add_edge_to_layer(edge, "ui")
        
        self.layered_graph = layered_graph
        return layered_graph
    
    async def analyze_graph(self) -> Dict[str, Any]:
        """
        Analyze the layered knowledge graph and return insights.
        
        Returns:
            A dictionary containing analysis results
        """
        if not self.layered_graph:
            raise ValueError("Layered graph has not been built yet")
        
        results = {}
        
        # Get metrics per layer
        results["layer_metrics"] = {}
        for layer in self.layered_graph.layers:
            layer_graph = self.layered_graph.get_layer_graph(layer.id)
            cumulative_graph = self.layered_graph.get_cumulative_layer_graph(layer.id)
            
            # Calculate metrics
            results["layer_metrics"][layer.id] = {
                "name": layer.name,
                "node_count": len(layer_graph.nodes),
                "edge_count": len(layer_graph.edges),
                "cumulative_node_count": len(cumulative_graph.nodes),
                "cumulative_edge_count": len(cumulative_graph.edges),
                "node_types": self._count_node_types(layer_graph.nodes),
                "relationship_types": self._count_relationship_types(layer_graph.edges)
            }
        
        # Calculate cross-layer relationships
        results["cross_layer_relationships"] = await self._analyze_cross_layer_relationships()
        
        # Calculate layer dependencies
        results["layer_dependencies"] = await self._analyze_layer_dependencies()
        
        # Identify critical components
        results["critical_components"] = await self._identify_critical_components()
        
        return results
    
    async def _analyze_cross_layer_relationships(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze relationships between different layers.
        
        Returns:
            A dictionary mapping each layer to its connections with other layers
        """
        cross_layer = {}
        
        # Analyze each layer's connections to nodes in other layers
        for source_layer in self.layered_graph.layers:
            cross_layer[source_layer.id] = {}
            
            source_nodes = {node.id for node in self.layered_graph.get_layer_graph(source_layer.id).nodes}
            
            for target_layer in self.layered_graph.layers:
                if source_layer.id == target_layer.id:
                    continue
                
                target_nodes = {node.id for node in self.layered_graph.get_layer_graph(target_layer.id).nodes}
                
                # Count connections from source layer to target layer
                connection_count = 0
                # Get all edges from cumulative graph for the source layer
                cumulative_graph = self.layered_graph.get_cumulative_layer_graph(source_layer.id)
                for edge in cumulative_graph.edges:
                    if (edge.source_node_id in source_nodes and 
                        edge.target_node_id in target_nodes):
                        connection_count += 1
                
                cross_layer[source_layer.id][target_layer.id] = connection_count
        
        return cross_layer
    
    async def _analyze_layer_dependencies(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Analyze dependencies between layers.
        
        Returns:
            A dictionary mapping each layer to its dependencies
        """
        dependencies = {}
        
        for layer in self.layered_graph.layers:
            dependencies[layer.id] = []
            
            for parent_id in layer.parent_layers:
                parent = next((l for l in self.layered_graph.layers if l.id == parent_id), None)
                if parent:
                    dependencies[layer.id].append({
                        "id": parent.id,
                        "name": parent.name,
                        "type": parent.layer_type
                    })
        
        return dependencies
    
    async def _identify_critical_components(self) -> List[Dict[str, Any]]:
        """
        Identify critical components based on connection count and cross-layer impact.
        
        Returns:
            A list of critical components with their metrics
        """
        critical_components = []
        
        # Count all connections for each node
        node_connections = {}
        node_layer_mapping = {}
        node_objects = {}
        
        # Map nodes to their layers and collect node objects
        for layer in self.layered_graph.layers:
            layer_graph = self.layered_graph.get_layer_graph(layer.id)
            for node in layer_graph.nodes:
                node_layer_mapping[node.id] = layer.id
                node_objects[node.id] = node
        
        # Get the cumulative graph for all layers to analyze all connections
        full_graph = self.layered_graph.get_cumulative_layer_graph(
            self.layered_graph.layers[-1].id  # Use the last layer to get everything
        )
        
        # Count connections
        for edge in full_graph.edges:
            # Count outgoing connections
            if edge.source_node_id not in node_connections:
                node_connections[edge.source_node_id] = {"incoming": 0, "outgoing": 0, "cross_layer": 0}
            node_connections[edge.source_node_id]["outgoing"] += 1
            
            # Count incoming connections
            if edge.target_node_id not in node_connections:
                node_connections[edge.target_node_id] = {"incoming": 0, "outgoing": 0, "cross_layer": 0}
            node_connections[edge.target_node_id]["incoming"] += 1
            
            # Check if this is a cross-layer connection
            if (edge.source_node_id in node_layer_mapping and 
                edge.target_node_id in node_layer_mapping and
                node_layer_mapping[edge.source_node_id] != node_layer_mapping[edge.target_node_id]):
                node_connections[edge.source_node_id]["cross_layer"] += 1
                node_connections[edge.target_node_id]["cross_layer"] += 1
        
        # Find nodes with high connection counts
        for node_id, connections in node_connections.items():
            total_connections = connections["incoming"] + connections["outgoing"]
            
            if total_connections > 3 or connections["cross_layer"] > 1:
                # Get the node object from our collected objects
                node = node_objects.get(node_id)
                
                if node:
                    critical_components.append({
                        "id": node.id,
                        "name": node.name,
                        "type": node.type,
                        "layer": node_layer_mapping.get(node.id, "unknown"),
                        "incoming_connections": connections["incoming"],
                        "outgoing_connections": connections["outgoing"],
                        "cross_layer_connections": connections["cross_layer"],
                        "total_connections": total_connections
                    })
        
        # Sort by total connections
        critical_components.sort(key=lambda x: x["total_connections"], reverse=True)
        
        return critical_components
    
    def _count_node_types(self, nodes: List[Node]) -> Dict[str, int]:
        """Count the frequency of each node type"""
        type_counts = {}
        for node in nodes:
            if node.type not in type_counts:
                type_counts[node.type] = 0
            type_counts[node.type] += 1
        return type_counts
    
    def _count_relationship_types(self, edges: List[Edge]) -> Dict[str, int]:
        """Count the frequency of each relationship type"""
        rel_counts = {}
        for edge in edges:
            if edge.relationship_name not in rel_counts:
                rel_counts[edge.relationship_name] = 0
            rel_counts[edge.relationship_name] += 1
        return rel_counts
    
    async def export_graph_visualization(self) -> Dict[str, Any]:
        """
        Export graph data in a format suitable for visualization.
        
        Returns:
            Dictionary with formatted graph data for visualization
        """
        if not self.layered_graph:
            raise ValueError("Layered graph has not been built yet")
        
        visualization_data = {
            "name": self.layered_graph.name,
            "description": self.layered_graph.description,
            "layers": [],
            "nodes": [],
            "edges": []
        }
        
        # Add layers
        for layer in self.layered_graph.layers:
            visualization_data["layers"].append({
                "id": layer.id,
                "name": layer.name,
                "description": layer.description,
                "type": layer.layer_type,
                "parent_layers": layer.parent_layers
            })
        
        # Add all nodes with their layer information
        for layer in self.layered_graph.layers:
            layer_graph = self.layered_graph.get_layer_graph(layer.id)
            for node in layer_graph.nodes:
                visualization_data["nodes"].append({
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "description": node.description,
                    "layer": layer.id
                })
        
        # Add all edges with their layer information
        for layer in self.layered_graph.layers:
            layer_graph = self.layered_graph.get_layer_graph(layer.id)
            for edge in layer_graph.edges:
                visualization_data["edges"].append({
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                    "relationship": edge.relationship_name,
                    "layer": layer.id
                })
        
        return visualization_data
    
    async def generate_report(self) -> str:
        """
        Generate a detailed report about the layered knowledge graph.
        
        Returns:
            A formatted report string
        """
        if not self.layered_graph:
            raise ValueError("Layered graph has not been built yet")
        
        analysis_results = await self.analyze_graph()
        
        report = []
        report.append("=" * 80)
        report.append(f"LAYERED KNOWLEDGE GRAPH REPORT: {self.layered_graph.name}")
        report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Overview
        report.append("\n1. OVERVIEW")
        report.append("-" * 40)
        report.append(f"Description: {self.layered_graph.description}")
        report.append(f"Number of layers: {len(self.layered_graph.layers)}")
        
        # Get node and edge counts for reporting
        total_nodes = sum(len(self.layered_graph.get_layer_graph(layer.id).nodes) for layer in self.layered_graph.layers)
        total_edges = sum(len(self.layered_graph.get_layer_graph(layer.id).edges) for layer in self.layered_graph.layers)
        
        report.append(f"Total nodes across all layers: {total_nodes}")
        report.append(f"Total edges across all layers: {total_edges}")
        
        # Get the cumulative graph for the last layer to get all nodes and edges
        if self.layered_graph.layers:
            final_cumulative = self.layered_graph.get_cumulative_layer_graph(self.layered_graph.layers[-1].id)
            report.append(f"Unique nodes in full graph: {len(final_cumulative.nodes)}")
            report.append(f"Unique edges in full graph: {len(final_cumulative.edges)}")
        
        # Layer Details
        report.append("\n2. LAYER DETAILS")
        report.append("-" * 40)
        
        for layer in self.layered_graph.layers:
            metrics = analysis_results["layer_metrics"][layer.id]
            parent_layers = ", ".join(layer.parent_layers) if layer.parent_layers else "None"
            
            report.append(f"\nLayer: {layer.name} (ID: {layer.id}, Type: {layer.layer_type})")
            report.append(f"Description: {layer.description}")
            report.append(f"Parent Layers: {parent_layers}")
            report.append(f"Nodes: {metrics['node_count']} (Cumulative: {metrics['cumulative_node_count']})")
            report.append(f"Edges: {metrics['edge_count']} (Cumulative: {metrics['cumulative_edge_count']})")
            
            report.append("Node Types:")
            for node_type, count in metrics["node_types"].items():
                report.append(f"  - {node_type}: {count}")
            
            report.append("Relationship Types:")
            for rel_type, count in metrics["relationship_types"].items():
                report.append(f"  - {rel_type}: {count}")
        
        # Cross-Layer Relationships
        report.append("\n3. CROSS-LAYER RELATIONSHIPS")
        report.append("-" * 40)
        
        for source_layer_id, target_layers in analysis_results["cross_layer_relationships"].items():
            source_layer = next((l for l in self.layered_graph.layers if l.id == source_layer_id), None)
            if source_layer:
                report.append(f"\nConnections from {source_layer.name} (ID: {source_layer_id}):")
                
                for target_layer_id, count in target_layers.items():
                    if count > 0:
                        target_layer = next((l for l in self.layered_graph.layers if l.id == target_layer_id), None)
                        if target_layer:
                            report.append(f"  - To {target_layer.name} (ID: {target_layer_id}): {count} connections")
        
        # Critical Components
        report.append("\n4. CRITICAL COMPONENTS")
        report.append("-" * 40)
        
        for component in analysis_results["critical_components"][:10]:  # Top 10 critical components
            layer_name = "Unknown"
            layer = next((l for l in self.layered_graph.layers if l.id == component["layer"]), None)
            if layer:
                layer_name = layer.name
                
            report.append(f"\n{component['name']} (ID: {component['id']}, Type: {component['type']})")
            report.append(f"Layer: {layer_name}")
            report.append(f"Total Connections: {component['total_connections']}")
            report.append(f"  - Incoming: {component['incoming_connections']}")
            report.append(f"  - Outgoing: {component['outgoing_connections']}")
            report.append(f"  - Cross-Layer: {component['cross_layer_connections']}")
        
        # Layer Dependencies
        report.append("\n5. LAYER DEPENDENCIES")
        report.append("-" * 40)
        
        for layer_id, dependencies in analysis_results["layer_dependencies"].items():
            layer = next((l for l in self.layered_graph.layers if l.id == layer_id), None)
            if layer:
                report.append(f"\n{layer.name} (ID: {layer_id}) depends on:")
                
                if not dependencies:
                    report.append("  - No dependencies (base layer)")
                else:
                    for dep in dependencies:
                        report.append(f"  - {dep['name']} (ID: {dep['id']}, Type: {dep['type']})")
        
        # Join all report lines into a single string
        return "\n".join(report)


async def run_advanced_demo():
    """Run the advanced layered graph demonstration"""
    logger.info("=== Starting Advanced Layered Graph Demonstration ===")
    
    try:
        demo = AdvancedLayeredGraphDemo()
        
        # Build the layered graph
        logger.info("Building layered knowledge graph...")
        layered_graph = await demo.build_layered_knowledge_graph()
        
        # Get node and edge counts for reporting
        total_nodes = sum(len(demo.layered_graph.get_layer_graph(layer.id).nodes) for layer in demo.layered_graph.layers)
        total_edges = sum(len(demo.layered_graph.get_layer_graph(layer.id).edges) for layer in demo.layered_graph.layers)
        
        logger.info(f"Built graph with {len(layered_graph.layers)} layers")
        logger.info(f"Total nodes across all layers: {total_nodes}")
        logger.info(f"Total edges across all layers: {total_edges}")
        
        # Get the cumulative graph for the last layer
        if layered_graph.layers:
            final_cumulative = layered_graph.get_cumulative_layer_graph(layered_graph.layers[-1].id)
            logger.info(f"Unique nodes in full graph: {len(final_cumulative.nodes)}")
            logger.info(f"Unique edges in full graph: {len(final_cumulative.edges)}")
        
        # Analyze the graph
        logger.info("\nAnalyzing layered graph...")
        analysis_results = await demo.analyze_graph()
        
        # Print some analysis highlights
        logger.info("\nLayer metrics:")
        for layer_id, metrics in analysis_results["layer_metrics"].items():
            logger.info(f"  - {metrics['name']}: {metrics['node_count']} nodes, {metrics['edge_count']} edges")
        
        logger.info("\nCritical components:")
        for component in analysis_results["critical_components"][:5]:  # Top 5
            logger.info(f"  - {component['name']} ({component['type']}): {component['total_connections']} total connections")
        
        # Generate and print the report
        logger.info("\nGenerating comprehensive report...")
        report = await demo.generate_report()
        
        # Print just the overview section
        overview_lines = report.split("2. LAYER DETAILS")[0]
        logger.info(f"\nReport Overview:\n{overview_lines}")
        
        # Export visualization data
        logger.info("\nExporting visualization data...")
        viz_data = await demo.export_graph_visualization()
        logger.info(f"Exported visualization data with {len(viz_data['nodes'])} nodes and {len(viz_data['edges'])} edges")
        
        # Save report to file
        report_filename = "layered_graph_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        logger.info(f"Full report saved to {report_filename}")
        
        # Save visualization data to file
        viz_filename = "layered_graph_visualization.json"
        with open(viz_filename, "w") as f:
            json.dump(viz_data, f, indent=2)
        logger.info(f"Visualization data saved to {viz_filename}")
        
        logger.info("\n=== Advanced Layered Graph Demonstration Completed ===")
        
    except Exception as e:
        logger.error(f"Error in advanced demo: {str(e)}")
        raise


async def main():
    """Main function"""
    await run_advanced_demo()


if __name__ == "__main__":
    asyncio.run(main()) 