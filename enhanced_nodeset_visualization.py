import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
import webbrowser
from pathlib import Path

import cognee
from cognee.shared.logging_utils import get_logger

# Configure logger
logger = get_logger(name="enhanced_graph_visualization")

# Type aliases for clarity
NodeData = Dict[str, Any]
EdgeData = Dict[str, Any]
GraphData = Tuple[List[Tuple[Any, Dict]], List[Tuple[Any, Any, Optional[str], Dict]]]


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class NodeSetVisualizer:
    """Class to create enhanced visualizations for NodeSet data in knowledge graphs."""
    
    # Color mapping for different node types
    NODE_COLORS = {
        "Entity": "#f47710",
        "EntityType": "#6510f4",
        "DocumentChunk": "#801212",
        "TextDocument": "#a83232",  # Darker red for documents
        "TextSummary": "#1077f4",
        "NodeSet": "#ff00ff",      # Bright magenta for NodeSet nodes
        "Unknown": "#999999",
        "default": "#D3D3D3",
    }
    
    # Size mapping for different node types
    NODE_SIZES = {
        "NodeSet": 20,              # Larger size for NodeSet nodes
        "TextDocument": 18,         # Larger size for document nodes
        "DocumentChunk": 18,        # Larger size for document nodes
        "TextSummary": 16,          # Medium size for TextSummary nodes
        "default": 13,              # Default size
    }
    
    def __init__(self):
        """Initialize the visualizer."""
        self.graph_engine = None
        self.nodes_data = []
        self.edges_data = []
        self.node_count = 0
        self.edge_count = 0
        self.nodeset_count = 0
        
    async def get_graph_data(self) -> bool:
        """Fetch graph data from the graph engine.
        
        Returns:
            bool: True if data was successfully retrieved, False otherwise.
        """
        self.graph_engine = await cognee.infrastructure.databases.graph.get_graph_engine()
        
        # Check if the graph exists and has nodes
        self.node_count = len(self.graph_engine.graph.nodes())
        self.edge_count = len(self.graph_engine.graph.edges())
        logger.info(f"Graph contains {self.node_count} nodes and {self.edge_count} edges")
        print(f"Graph contains {self.node_count} nodes and {self.edge_count} edges")
        
        if self.node_count == 0:
            logger.error("The graph is empty! Please run a test script first to generate data.")
            print("ERROR: The graph is empty! Please run a test script first to generate data.")
            return False
        
        graph_data = await self.graph_engine.get_graph_data()
        self.nodes_data, self.edges_data = graph_data
        
        # Count NodeSets for status display
        self.nodeset_count = sum(1 for _, info in self.nodes_data if info.get("type") == "NodeSet")
        
        return True
    
    def prepare_node_data(self) -> List[NodeData]:
        """Process raw node data to prepare for visualization.
        
        Returns:
            List[NodeData]: List of prepared node data objects.
        """
        nodes_list = []
        
        # Create a lookup for node types for faster access
        node_type_lookup = {str(node_id): node_info.get("type", "Unknown") 
                           for node_id, node_info in self.nodes_data}
        
        for node_id, node_info in self.nodes_data:
            # Create a clean copy to avoid modifying the original
            processed_node = node_info.copy()
            
            # Remove fields that cause JSON serialization issues
            self._clean_node_data(processed_node)
            
            # Add required visualization properties
            processed_node["id"] = str(node_id)
            node_type = processed_node.get("type", "default")
            
            # Apply visual styling based on node type
            processed_node["color"] = self.NODE_COLORS.get(node_type, self.NODE_COLORS["default"])
            processed_node["size"] = self.NODE_SIZES.get(node_type, self.NODE_SIZES["default"])
            
            # Create display names
            self._format_node_display_name(processed_node, node_type)
            
            nodes_list.append(processed_node)
            
        return nodes_list
    
    @staticmethod
    def _clean_node_data(node: NodeData) -> None:
        """Remove fields that might cause JSON serialization issues.
        
        Args:
            node: The node data to clean
        """
        # Remove non-essential fields that might cause serialization issues
        for key in ["created_at", "updated_at", "raw_data_location"]:
            if key in node:
                del node[key]
    
    @staticmethod
    def _format_node_display_name(node: NodeData, node_type: str) -> None:
        """Format the display name for a node.
        
        Args:
            node: The node data to process
            node_type: The type of the node
        """
        # Set a default name if none exists
        node["name"] = node.get("name", node.get("id", "Unknown"))
        
        # Special formatting for NodeSet nodes
        if node_type == "NodeSet" and "node_id" in node:
            node["display_name"] = f"NodeSet: {node['node_id']}"
        else:
            node["display_name"] = node["name"]
            
        # Truncate long display names
        if len(node["display_name"]) > 30:
            node["display_name"] = f"{node['display_name'][:27]}..."
    
    def prepare_edge_data(self, nodes_list: List[NodeData]) -> List[EdgeData]:
        """Process raw edge data to prepare for visualization.
        
        Args:
            nodes_list: The processed node data
            
        Returns:
            List[EdgeData]: List of prepared edge data objects.
        """
        links_list = []
        
        # Create a lookup for node types for faster access
        node_type_lookup = {node["id"]: node.get("type", "Unknown") for node in nodes_list}
        
        for source, target, relation, edge_info in self.edges_data:
            source_str = str(source)
            target_str = str(target)
            
            # Skip if source or target not in node_type_lookup (should not happen)
            if source_str not in node_type_lookup or target_str not in node_type_lookup:
                continue
            
            # Get node types
            source_type = node_type_lookup[source_str]
            target_type = node_type_lookup[target_str]
            
            # Create edge data
            link_data = {
                "source": source_str,
                "target": target_str,
                "relation": relation or "UNKNOWN"
            }
            
            # Categorize the edge for styling
            link_data["connection_type"] = self._categorize_edge(source_type, target_type)
            
            links_list.append(link_data)
            
        return links_list
    
    @staticmethod
    def _categorize_edge(source_type: str, target_type: str) -> str:
        """Categorize an edge based on the connected node types.
        
        Args:
            source_type: The type of the source node
            target_type: The type of the target node
            
        Returns:
            str: The category of the edge
        """
        if source_type == "NodeSet" and target_type != "NodeSet":
            return "nodeset_to_value"
        elif (source_type in ["TextDocument", "DocumentChunk", "TextSummary"]) and target_type == "NodeSet":
            return "document_to_nodeset"
        elif target_type == "NodeSet":
            return "to_nodeset"
        elif source_type == "NodeSet":
            return "from_nodeset"
        else:
            return "standard"
    
    def generate_html(self, nodes_list: List[NodeData], links_list: List[EdgeData]) -> str:
        """Generate the HTML visualization with D3.js.
        
        Args:
            nodes_list: The processed node data
            links_list: The processed edge data
            
        Returns:
            str: The HTML content for the visualization
        """
        # Use embedded template directly - more reliable than file access
        html_template = self._get_embedded_html_template()
        
        # Generate the HTML content with custom JSON encoder for datetime objects
        html_content = html_template.replace("{nodes}", json.dumps(nodes_list, cls=DateTimeEncoder))
        html_content = html_content.replace("{links}", json.dumps(links_list, cls=DateTimeEncoder))
        html_content = html_content.replace("{node_count}", str(self.node_count))
        html_content = html_content.replace("{edge_count}", str(self.edge_count))
        html_content = html_content.replace("{nodeset_count}", str(self.nodeset_count))
        
        return html_content
    
    def save_html(self, html_content: str) -> str:
        """Save the HTML content to a file and open it in the browser.
        
        Args:
            html_content: The HTML content to save
            
        Returns:
            str: The path to the saved file
        """
        # Create the output file path
        output_path = Path.cwd() / "enhanced_nodeset_visualization.html"
        
        # Write the HTML content to the file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Enhanced visualization saved to: {output_path}")
        print(f"Enhanced visualization saved to: {output_path}")
        
        # Open the visualization in the default web browser
        file_url = f"file://{output_path}"
        logger.info(f"Opening visualization in browser: {file_url}")
        print(f"Opening enhanced visualization in browser: {file_url}")
        webbrowser.open(file_url)
        
        return str(output_path)
    
    @staticmethod
    def _get_embedded_html_template() -> str:
        """Get the embedded HTML template as a fallback.
        
        Returns:
            str: The HTML template
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Cognee NodeSet Visualization</title>
            <script src="https://d3js.org/d3.v5.min.js"></script>
            <style>
                body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: linear-gradient(90deg, #101010, #1a1a2e); color: white; font-family: 'Inter', Arial, sans-serif; }

                svg { width: 100vw; height: 100vh; display: block; }
                .links line { stroke-width: 2px; }
                .nodes circle { stroke: white; stroke-width: 0.5px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
                .node-label { font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', Arial, sans-serif; pointer-events: none; }
                .edge-label { font-size: 3px; fill: rgba(255, 255, 255, 0.7); text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', Arial, sans-serif; pointer-events: none; }
                
                /* NodeSet specific styles */
                .links line.nodeset_to_value { stroke: #ff00ff; stroke-width: 3px; stroke-dasharray: 5, 5; }
                .links line.document_to_nodeset { stroke: #fc0; stroke-width: 3px; }
                .links line.to_nodeset { stroke: #0cf; stroke-width: 2px; }
                .links line.from_nodeset { stroke: #0fc; stroke-width: 2px; }
                
                .nodes circle.nodeset { stroke: white; stroke-width: 2px; filter: drop-shadow(0 0 10px rgba(255,0,255,0.8)); }
                .nodes circle.document { stroke: white; stroke-width: 1.5px; filter: drop-shadow(0 0 8px rgba(255,255,0,0.6)); }
                .node-label.nodeset { font-size: 6px; font-weight: bold; fill: white; }
                .node-label.document { font-size: 5.5px; font-weight: bold; fill: white; }
                
                /* Legend */
                .legend { position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }
                .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
                .legend-color { width: 15px; height: 15px; margin-right: 10px; border-radius: 50%; }
                .legend-label { font-size: 14px; }
                
                /* Edge legend */
                .edge-legend { position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }
                .edge-legend-item { display: flex; align-items: center; margin-bottom: 10px; }
                .edge-line { width: 30px; height: 3px; margin-right: 10px; }
                .edge-label { font-size: 14px; }
                
                /* Controls */
                .controls { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; display: flex; gap: 10px; }
                button { background: #333; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; }
                button:hover { background: #555; }
                
                /* Status message */
                .status { position: fixed; bottom: 10px; right: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <svg></svg>
            
            <!-- Node Legend -->
            <div class="legend">
                <h3 style="margin-top: 0;">Node Types</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff00ff;"></div>
                    <div class="legend-label">NodeSet</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #a83232;"></div>
                    <div class="legend-label">TextDocument</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #801212;"></div>
                    <div class="legend-label">DocumentChunk</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #1077f4;"></div>
                    <div class="legend-label">TextSummary</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f47710;"></div>
                    <div class="legend-label">Entity</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #6510f4;"></div>
                    <div class="legend-label">EntityType</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #999999;"></div>
                    <div class="legend-label">Unknown</div>
                </div>
            </div>
            
            <!-- Edge Legend -->
            <div class="edge-legend">
                <h3 style="margin-top: 0;">Edge Types</h3>
                <div class="edge-legend-item">
                    <div class="edge-line" style="background-color: #fc0;"></div>
                    <div class="edge-label">Document → NodeSet</div>
                </div>
                <div class="edge-legend-item">
                    <div class="edge-line" style="background-color: #ff00ff; height: 3px; background: linear-gradient(to right, #ff00ff 50%, transparent 50%); background-size: 10px 3px; background-repeat: repeat-x;"></div>
                    <div class="edge-label">NodeSet → Value</div>
                </div>
                <div class="edge-legend-item">
                    <div class="edge-line" style="background-color: #0cf;"></div>
                    <div class="edge-label">Any → NodeSet</div>
                </div>
                <div class="edge-legend-item">
                    <div class="edge-line" style="background-color: rgba(255, 255, 255, 0.4);"></div>
                    <div class="edge-label">Standard Connection</div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button id="center-btn">Center Graph</button>
                <button id="highlight-nodesets">Highlight NodeSets</button>
                <button id="highlight-documents">Highlight Documents</button>
                <button id="reset-highlight">Reset Highlight</button>
            </div>
            
            <!-- Status -->
            <div class="status">
                <div>Nodes: {node_count}</div>
                <div>Edges: {edge_count}</div>
                <div>NodeSets: {nodeset_count}</div>
            </div>
            
            <script>
                var nodes = {nodes};
                var links = {links};

                var svg = d3.select("svg"),
                    width = window.innerWidth,
                    height = window.innerHeight;

                var container = svg.append("g");
                
                // Count NodeSets for status display
                const nodesetCount = nodes.filter(n => n.type === "NodeSet").length;
                document.querySelector('.status').innerHTML = `
                    <div>Nodes: ${nodes.length}</div>
                    <div>Edges: ${links.length}</div>
                    <div>NodeSets: ${nodesetCount}</div>
                `;

                var simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).strength(0.1))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("x", d3.forceX().strength(0.1).x(width / 2))
                    .force("y", d3.forceY().strength(0.1).y(height / 2));

                var link = container.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .attr("stroke-width", d => {
                        if (d.connection_type === 'document_to_nodeset' || d.connection_type === 'nodeset_to_value') {
                            return 3;
                        }
                        return 2;
                    })
                    .attr("stroke", d => {
                        switch(d.connection_type) {
                            case 'document_to_nodeset': return "#fc0";
                            case 'nodeset_to_value': return "#ff00ff";
                            case 'to_nodeset': return "#0cf";
                            case 'from_nodeset': return "#0fc";
                            default: return "rgba(255, 255, 255, 0.4)";
                        }
                    })
                    .attr("stroke-dasharray", d => d.connection_type === 'nodeset_to_value' ? "5,5" : null)
                    .attr("class", d => d.connection_type);

                var edgeLabels = container.append("g")
                    .attr("class", "edge-labels")
                    .selectAll("text")
                    .data(links)
                    .enter().append("text")
                    .attr("class", "edge-label")
                    .text(d => d.relation);

                var nodeGroup = container.append("g")
                    .attr("class", "nodes")
                    .selectAll("g")
                    .data(nodes)
                    .enter().append("g");

                var node = nodeGroup.append("circle")
                    .attr("r", d => d.size || 13)
                    .attr("fill", d => d.color)
                    .attr("class", d => {
                        if (d.type === "NodeSet") return "nodeset";
                        if (d.type === "TextDocument" || d.type === "DocumentChunk") return "document";
                        return "";
                    })
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));

                nodeGroup.append("text")
                    .attr("class", d => {
                        if (d.type === "NodeSet") return "node-label nodeset";
                        if (d.type === "TextDocument" || d.type === "DocumentChunk") return "node-label document";
                        return "node-label";
                    })
                    .attr("dy", 4)
                    .attr("font-size", d => {
                        if (d.type === "NodeSet") return "6px";
                        if (d.type === "TextDocument" || d.type === "DocumentChunk") return "5.5px";
                        return "5px";
                    })
                    .attr("text-anchor", "middle")
                    .text(d => d.display_name || d.name);

                node.append("title").text(d => {
                    // Create a formatted tooltip with node properties
                    let props = Object.entries(d)
                        .filter(([key]) => !["x", "y", "vx", "vy", "index", "fx", "fy", "color", "display_name"].includes(key))
                        .map(([key, value]) => `${key}: ${value}`)
                        .join("\\n");
                    return props;
                });

                simulation.on("tick", function() {
                    link.attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);

                    edgeLabels
                        .attr("x", d => (d.source.x + d.target.x) / 2)
                        .attr("y", d => (d.source.y + d.target.y) / 2 - 5);

                    node.attr("cx", d => d.x)
                        .attr("cy", d => d.y);

                    nodeGroup.select("text")
                        .attr("x", d => d.x)
                        .attr("y", d => d.y)
                        .attr("dy", 4)
                        .attr("text-anchor", "middle");
                });

                // Add zoom behavior
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 8])
                    .on("zoom", function() {
                        container.attr("transform", d3.event.transform);
                    });
                    
                svg.call(zoom);
                
                // Button controls
                document.getElementById("center-btn").addEventListener("click", function() {
                    svg.transition().duration(750).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(width / 2, height / 2).scale(1)
                    );
                });
                
                document.getElementById("highlight-nodesets").addEventListener("click", function() {
                    highlightNodes("NodeSet");
                });
                
                document.getElementById("highlight-documents").addEventListener("click", function() {
                    highlightNodes(["TextDocument", "DocumentChunk"]);
                });
                
                document.getElementById("reset-highlight").addEventListener("click", function() {
                    resetHighlight();
                });
                
                function highlightNodes(typeToHighlight) {
                    // Dim all nodes and links
                    node.transition().duration(300)
                        .attr("opacity", 0.2);
                    link.transition().duration(300)
                        .attr("opacity", 0.2);
                    nodeGroup.selectAll("text").transition().duration(300)
                        .attr("opacity", 0.2);
                    
                    // Create arrays for types if a single string is provided
                    const typesToHighlight = Array.isArray(typeToHighlight) ? typeToHighlight : [typeToHighlight];
                    
                    // Highlight matching nodes and their connected nodes
                    const highlightedNodeIds = new Set();
                    
                    // First, find all nodes of the target type
                    nodes.forEach(n => {
                        if (typesToHighlight.includes(n.type)) {
                            highlightedNodeIds.add(n.id);
                        }
                    });
                    
                    // Find all connected nodes (both directions)
                    links.forEach(l => {
                        if (highlightedNodeIds.has(l.source.id || l.source)) {
                            highlightedNodeIds.add(l.target.id || l.target);
                        }
                        if (highlightedNodeIds.has(l.target.id || l.target)) {
                            highlightedNodeIds.add(l.source.id || l.source);
                        }
                    });
                    
                    // Highlight the nodes
                    node.filter(d => highlightedNodeIds.has(d.id))
                        .transition().duration(300)
                        .attr("opacity", 1);
                    
                    // Highlight the labels
                    nodeGroup.selectAll("text")
                        .filter(d => highlightedNodeIds.has(d.id))
                        .transition().duration(300)
                        .attr("opacity", 1);
                    
                    // Highlight the links between highlighted nodes
                    link.filter(d => {
                        const sourceId = d.source.id || d.source;
                        const targetId = d.target.id || d.target;
                        return highlightedNodeIds.has(sourceId) && highlightedNodeIds.has(targetId);
                    })
                    .transition().duration(300)
                    .attr("opacity", 1);
                }
                
                function resetHighlight() {
                    node.transition().duration(300).attr("opacity", 1);
                    link.transition().duration(300).attr("opacity", 1);
                    nodeGroup.selectAll("text").transition().duration(300).attr("opacity", 1);
                }

                function dragstarted(d) {
                    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }

                function dragged(d) {
                    d.fx = d3.event.x;
                    d.fy = d3.event.y;
                }

                function dragended(d) {
                    if (!d3.event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }

                window.addEventListener("resize", function() {
                    width = window.innerWidth;
                    height = window.innerHeight;
                    svg.attr("width", width).attr("height", height);
                    simulation.force("center", d3.forceCenter(width / 2, height / 2));
                    simulation.alpha(1).restart();
                });
            </script>
        </body>
        </html>
        """
    
    async def create_visualization(self) -> Optional[str]:
        """Main method to create the visualization.
        
        Returns:
            Optional[str]: Path to the saved visualization file, or None if unsuccessful
        """
        print("Creating enhanced NodeSet visualization...")
        
        # Get graph data
        if not await self.get_graph_data():
            return None
        
        try:
            # Process nodes
            nodes_list = self.prepare_node_data()
            
            # Process edges
            links_list = self.prepare_edge_data(nodes_list)
            
            # Generate HTML
            html_content = self.generate_html(nodes_list, links_list)
            
            # Save to file and open in browser
            return self.save_html(html_content)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            print(f"Error creating visualization: {str(e)}")
            return None


async def main():
    """Main entry point for the script."""
    visualizer = NodeSetVisualizer()
    await visualizer.create_visualization()


if __name__ == "__main__":
    # Run the async main function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close() 