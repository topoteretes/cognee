# Quickstart

> Get started with Cognee quickly and efficiently

export const GraphVisualization = () => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  useEffect(() => {
    const loadD3 = async () => {
      if (typeof window === 'undefined') return;
      if (!window.d3) {
        const script = document.createElement('script');
        script.src = 'https://d3js.org/d3.v5.min.js';
        script.async = true;
        document.body.appendChild(script);
        await new Promise(resolve => {
          script.onload = resolve;
        });
      }
      initializeGraph();
    };
    const initializeGraph = () => {
      const d3 = window.d3;
      if (!d3 || !svgRef.current || !tooltipRef.current) return;
      const nodes = [{
        "name": "",
        "type": "DocumentChunk",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["text"]
        },
        "belongs_to_set": null,
        "text": "Cognee turns documents into AI memory.",
        "chunk_size": 14,
        "chunk_index": 0,
        "cut_type": "sentence_end",
        "id": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "color": "#801212"
      }, {
        "name": "cognee",
        "type": "Entity",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "description": "Tool that turns documents into AI memory.",
        "id": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
        "color": "#f47710"
      }, {
        "name": "product",
        "type": "EntityType",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "description": "product",
        "id": "379107f4-bd9d-5823-990f-cc4becfeccfc",
        "color": "#6510f4"
      }, {
        "name": "documents",
        "type": "Entity",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "description": "Source materials processed by Cognee.",
        "id": "f4653eeb-de7f-51f4-b5ab-516f8d3bf5e7",
        "color": "#f47710"
      }, {
        "name": "concept",
        "type": "EntityType",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "description": "concept",
        "id": "dd9713b7-dc20-5101-aad0-1c4216811147",
        "color": "#6510f4"
      }, {
        "name": "ai memory",
        "type": "Entity",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "description": "Representation of documents as memory usable by AI.",
        "id": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
        "color": "#f47710"
      }, {
        "name": "text_document",
        "type": "TextDocument",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["name"]
        },
        "belongs_to_set": null,
        "raw_data_location": "",
        "external_metadata": "{}",
        "mime_type": "text/plain",
        "id": "d08449e0-0ce8-52e9-97ab-44e1e8efc4a7",
        "color": "#D3D3D3"
      }, {
        "name": "",
        "type": "TextSummary",
        "ontology_valid": false,
        "version": 1,
        "topological_rank": 0,
        "metadata": {
          "index_fields": ["text"]
        },
        "belongs_to_set": null,
        "text": "Cognee transforms documents into AI memory.",
        "id": "59c04417-3fad-53dd-a7b5-dd2ce28dd2dc",
        "color": "#1077f4"
      }];
      const links = [{
        "source": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "target": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
        "relation": "contains",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "9e42824c-569a-5b89-808f-8b8ece56476e",
          "target_node_id": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
          "relationship_name": "contains",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "target": "f4653eeb-de7f-51f4-b5ab-516f8d3bf5e7",
        "relation": "contains",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "9e42824c-569a-5b89-808f-8b8ece56476e",
          "target_node_id": "f4653eeb-de7f-51f4-b5ab-516f8d3bf5e7",
          "relationship_name": "contains",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "target": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
        "relation": "contains",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "9e42824c-569a-5b89-808f-8b8ece56476e",
          "target_node_id": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
          "relationship_name": "contains",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "target": "d08449e0-0ce8-52e9-97ab-44e1e8efc4a7",
        "relation": "is_part_of",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "9e42824c-569a-5b89-808f-8b8ece56476e",
          "target_node_id": "d08449e0-0ce8-52e9-97ab-44e1e8efc4a7",
          "relationship_name": "is_part_of",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
        "target": "379107f4-bd9d-5823-990f-cc4becfeccfc",
        "relation": "is_a",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
          "target_node_id": "379107f4-bd9d-5823-990f-cc4becfeccfc",
          "relationship_name": "is_a",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
        "target": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
        "relation": "converts_documents_into",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "relationship_name": "converts_documents_into",
          "source_node_id": "2d09f34b-aee1-5d32-a2f9-23eb7ed48b24",
          "target_node_id": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
          "ontology_valid": false
        }
      }, {
        "source": "f4653eeb-de7f-51f4-b5ab-516f8d3bf5e7",
        "target": "dd9713b7-dc20-5101-aad0-1c4216811147",
        "relation": "is_a",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "f4653eeb-de7f-51f4-b5ab-516f8d3bf5e7",
          "target_node_id": "dd9713b7-dc20-5101-aad0-1c4216811147",
          "relationship_name": "is_a",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
        "target": "dd9713b7-dc20-5101-aad0-1c4216811147",
        "relation": "is_a",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "af397680-8867-5ec5-9ba9-dddd39dfd7a8",
          "target_node_id": "dd9713b7-dc20-5101-aad0-1c4216811147",
          "relationship_name": "is_a",
          "updated_at": "2025-12-04 08:54:17"
        }
      }, {
        "source": "59c04417-3fad-53dd-a7b5-dd2ce28dd2dc",
        "target": "9e42824c-569a-5b89-808f-8b8ece56476e",
        "relation": "made_from",
        "weight": null,
        "all_weights": {},
        "relationship_type": null,
        "edge_info": {
          "source_node_id": "59c04417-3fad-53dd-a7b5-dd2ce28dd2dc",
          "target_node_id": "9e42824c-569a-5b89-808f-8b8ece56476e",
          "relationship_name": "made_from",
          "updated_at": "2025-12-04 08:54:17"
        }
      }];
      d3.select(svgRef.current).selectAll("*").remove();
      const svg = d3.select(svgRef.current);
      const width = 800;
      const height = 600;
      svg.attr("width", width).attr("height", height);
      const container = svg.append("g");
      const tooltip = d3.select(tooltipRef.current);
      const simulation = d3.forceSimulation(nodes).force("link", d3.forceLink(links).id(d => d.id).strength(0.1)).force("charge", d3.forceManyBody().strength(-275)).force("center", d3.forceCenter(width / 2, height / 2)).force("x", d3.forceX().strength(0.1).x(width / 2)).force("y", d3.forceY().strength(0.1).y(height / 2));
      const link = container.append("g").attr("class", "links").selectAll("line").data(links).enter().append("line").attr("stroke-width", d => {
        if (d.weight) return Math.max(2, d.weight * 5);
        if (d.all_weights && Object.keys(d.all_weights).length > 0) {
          const avgWeight = Object.values(d.all_weights).reduce((a, b) => a + b, 0) / Object.values(d.all_weights).length;
          return Math.max(2, avgWeight * 5);
        }
        return 2;
      }).attr("class", d => {
        if (d.all_weights && Object.keys(d.all_weights).length > 1) return "multi-weighted";
        if (d.weight || d.all_weights && Object.keys(d.all_weights).length > 0) return "weighted";
        return "";
      }).on("mouseover", function (d) {
        let content = "<strong>Edge Information</strong><br/>";
        content += "Relationship: " + d.relation + "<br/>";
        if (d.all_weights && Object.keys(d.all_weights).length > 0) {
          content += "<strong>Weights:</strong><br/>";
          Object.keys(d.all_weights).forEach(function (weightName) {
            content += "&nbsp;&nbsp;" + weightName + ": " + d.all_weights[weightName] + "<br/>";
          });
        } else if (d.weight !== null && d.weight !== undefined) {
          content += "Weight: " + d.weight + "<br/>";
        }
        if (d.relationship_type) {
          content += "Type: " + d.relationship_type + "<br/>";
        }
        if (d.edge_info) {
          Object.keys(d.edge_info).forEach(function (key) {
            if (key !== 'weight' && key !== 'weights' && key !== 'relationship_type' && key !== 'source_node_id' && key !== 'target_node_id' && key !== 'relationship_name' && key !== 'updated_at' && !key.startsWith('weight_')) {
              content += key + ": " + d.edge_info[key] + "<br/>";
            }
          });
        }
        tooltip.html(content).style("left", d3.event.pageX + 10 + "px").style("top", d3.event.pageY - 10 + "px").style("opacity", 1);
      }).on("mouseout", function () {
        tooltip.style("opacity", 0);
      });
      const edgeLabels = container.append("g").attr("class", "edge-labels").selectAll("text").data(links).enter().append("text").attr("class", "edge-label").text(d => {
        let label = d.relation;
        if (d.all_weights && Object.keys(d.all_weights).length > 1) {
          label += " (" + Object.keys(d.all_weights).length + " weights)";
        } else if (d.weight) {
          label += " (" + d.weight + ")";
        } else if (d.all_weights && Object.keys(d.all_weights).length === 1) {
          const singleWeight = Object.values(d.all_weights)[0];
          label += " (" + singleWeight + ")";
        }
        return label;
      });
      const nodeGroup = container.append("g").attr("class", "nodes").selectAll("g").data(nodes).enter().append("g");
      const node = nodeGroup.append("circle").attr("r", 13).attr("fill", d => d.color).call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));
      nodeGroup.append("text").attr("class", "node-label").attr("dy", 4).attr("text-anchor", "middle").text(d => d.name);
      node.append("title").text(d => JSON.stringify(d));
      simulation.on("tick", function () {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        edgeLabels.attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2 - 5);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        nodeGroup.select("text").attr("x", d => d.x).attr("y", d => d.y).attr("dy", 4).attr("text-anchor", "middle");
      });
      svg.call(d3.zoom().on("zoom", function () {
        container.attr("transform", d3.event.transform);
      }));
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
    };
    loadD3();
  }, []);
  return <div style={{
    position: 'relative',
    width: '100%',
    height: '600px'
  }}>
      <style>{`
        .links line {
          stroke: rgba(255, 255, 255, 0.4);
          stroke-width: 2px;
        }
        .links line.weighted {
          stroke: rgba(255, 215, 0, 0.7);
        }
        .links line.multi-weighted {
          stroke: rgba(0, 255, 127, 0.8);
        }
        .nodes circle {
          stroke: white;
          stroke-width: 0.5px;
          filter: drop-shadow(0 0 5px rgba(255,255,255,0.3));
        }
        .node-label {
          font-size: 5px;
          font-weight: bold;
          fill: white;
          text-anchor: middle;
          dominant-baseline: middle;
          font-family: 'Inter', sans-serif;
          pointer-events: none;
        }
        .edge-label {
          font-size: 3px;
          fill: rgba(255, 255, 255, 0.7);
          text-anchor: middle;
          dominant-baseline: middle;
          font-family: 'Inter', sans-serif;
          pointer-events: none;
        }
      `}</style>
      <svg ref={svgRef} style={{
    width: '100%',
    height: '100%',
    display: 'block',
    background: 'linear-gradient(90deg, #101010, #1a1a2e)'
  }} />
      <div ref={tooltipRef} style={{
    position: 'absolute',
    textAlign: 'left',
    padding: '8px',
    fontSize: '12px',
    background: 'rgba(0, 0, 0, 0.9)',
    color: 'white',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    borderRadius: '4px',
    pointerEvents: 'none',
    opacity: 0,
    transition: 'opacity 0.2s',
    zIndex: 1000,
    maxWidth: '300px',
    wordWrap: 'break-word'
  }} />
      <svg style={{
    position: 'absolute',
    bottom: '10px',
    right: '10px',
    width: '150px',
    height: 'auto',
    zIndex: 9999
  }} viewBox="0 0 158 44" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fillRule="evenodd" clipRule="evenodd" d="M11.7496 4.92654C7.83308 4.92654 4.8585 7.94279 4.8585 11.3612V14.9304C4.8585 18.3488 7.83308 21.3651 11.7496 21.3651C13.6831 21.3651 15.0217 20.8121 16.9551 19.3543C18.0458 18.5499 19.5331 18.8013 20.3263 19.9072C21.1195 21.0132 20.8717 22.5213 19.781 23.3257C17.3518 25.0851 15.0217 26.2414 11.7 26.2414C5.35425 26.2414 0 21.2646 0 14.9304V11.3612C0 4.97681 5.35425 0.0502739 11.7 0.0502739C15.0217 0.0502739 17.3518 1.2065 19.781 2.96598C20.8717 3.77032 21.1195 5.27843 20.3263 6.38439C19.5331 7.49035 18.0458 7.69144 16.9551 6.93737C15.0217 5.52979 13.6831 4.92654 11.7496 4.92654ZM35.5463 4.92654C31.7289 4.92654 28.6552 8.04333 28.6552 11.8639V14.478C28.6552 18.2986 31.7289 21.4154 35.5463 21.4154C39.3141 21.4154 42.3878 18.2986 42.3878 14.478V11.8639C42.3878 8.04333 39.3141 4.92654 35.5463 4.92654ZM23.7967 11.8639C23.7967 5.32871 29.0518 0 35.5463 0C42.0408 0 47.2463 5.32871 47.2463 11.8639V14.478C47.2463 21.0132 42.0408 26.3419 35.5463 26.3419C29.0518 26.3419 23.7967 21.0635 23.7967 14.478V11.8639ZM63.3091 5.07736C59.4917 5.07736 56.418 8.19415 56.418 12.0147C56.418 15.8353 59.4917 18.9521 63.3091 18.9521C67.1265 18.9521 70.1506 15.8856 70.1506 12.0147C70.1506 8.14388 67.0769 5.07736 63.3091 5.07736ZM51.5595 11.9645C51.5595 5.42925 56.8146 0.150814 63.3091 0.150814C66.0854 0.150814 68.5642 1.10596 70.5968 2.71463L72.4311 0.904876C73.3731 -0.0502693 74.9099 -0.0502693 75.8519 0.904876C76.7938 1.86002 76.7938 3.41841 75.8519 4.37356L73.7201 6.53521C74.5629 8.19414 75.0587 10.0542 75.0587 12.0147C75.0587 18.4997 69.8532 23.8284 63.3587 23.8284C63.3091 23.8284 63.2099 23.8284 63.1603 23.8284H58.0044C57.1616 23.8284 56.4675 24.5322 56.4675 25.3868C56.4675 26.2414 57.1616 26.9452 58.0044 26.9452H64.6476H66.7794C68.5146 26.9452 70.3489 27.4479 71.7866 28.6041C73.2739 29.8106 74.2159 31.5701 74.4142 33.7317C74.7116 37.6026 72.0345 40.2166 69.8532 41.0713L63.8048 43.7859C62.5654 44.3389 61.1277 43.7859 60.6319 42.5291C60.0866 41.2723 60.6319 39.8648 61.8714 39.3118L68.0188 36.5972C68.0684 36.5972 68.118 36.5469 68.1675 36.5469C68.4154 36.4463 68.8616 36.1447 69.2087 35.6923C69.5061 35.2398 69.7044 34.7371 69.6548 34.1339C69.6053 33.229 69.2582 32.7263 68.8616 32.4247C68.4154 32.0728 67.7214 31.8214 66.8786 31.8214H58.2027C58.1531 31.8214 58.1531 31.8214 58.1035 31.8214H58.054C54.534 31.8214 51.6586 28.956 51.6586 25.3868C51.6586 23.0743 52.8485 21.0635 54.6828 19.9072C52.6997 17.7959 51.5595 15.031 51.5595 11.9645ZM90.8736 5.07736C87.0562 5.07736 83.9824 8.19415 83.9824 12.0147V23.9289C83.9824 25.2862 82.8917 26.3922 81.5532 26.3922C80.2146 26.3922 79.1239 25.2862 79.1239 23.9289V11.9645C79.1239 5.42925 84.379 0.150814 90.824 0.150814C97.2689 0.150814 102.524 5.42925 102.524 11.9645V23.8786C102.524 25.2359 101.433 26.3419 100.095 26.3419C98.7562 26.3419 97.6655 25.2359 97.6655 23.8786V11.9645C97.7647 8.14387 94.6414 5.07736 90.8736 5.07736ZM119.43 5.07736C115.513 5.07736 112.39 8.24441 112.39 12.065V14.5785C112.39 18.4494 115.513 21.5662 119.43 21.5662C120.768 21.5662 122.057 21.164 123.098 20.5105C124.238 19.8067 125.726 20.1586 126.42 21.3148C127.114 22.4711 126.767 23.9792 125.627 24.683C123.842 25.7889 121.71 26.4425 119.43 26.4425C112.885 26.4425 107.581 21.1137 107.581 14.5785V12.065C107.581 5.47952 112.935 0.201088 119.43 0.201088C125.032 0.201088 129.692 4.07194 130.931 9.3001L131.427 11.3612L121.115 15.584C119.876 16.0867 118.488 15.4834 117.942 14.2266C117.447 12.9699 118.041 11.5623 119.281 11.0596L125.478 8.54604C124.238 6.43466 122.008 5.07736 119.43 5.07736ZM146.003 5.07736C142.086 5.07736 138.963 8.24441 138.963 12.065V14.5785C138.963 18.4494 142.086 21.5662 146.003 21.5662C147.341 21.5662 148.630 21.164 149.671 20.5105C150.217 20.1586 150.663 19.8067 151.109 19.304C152.001 18.2986 153.538 18.2483 154.53 19.2034C155.521 20.1083 155.571 21.6667 154.629 22.6721C153.935 23.4262 153.092 24.13 152.2 24.683C150.415 25.7889 148.283 26.4425 146.003 26.4425C139.458 26.4425 134.154 21.1137 134.154 14.5785V12.065C134.154 5.47952 139.508 0.201088 146.003 0.201088C151.605 0.201088 156.265 4.07194 157.504 9.3001L158 11.3612L147.688 15.584C146.449 16.0867 145.061 15.4834 144.515 14.2266C144.019 12.9699 144.614 11.5623 145.854 11.0596L152.051 8.54604C150.762 6.43466 148.58 5.07736 146.003 5.07736Z" fill="white" />
      </svg>
    </div>;
};

After completing the [installation steps](https://docs.cognee.ai/getting-started/installation) successfully, run your first Cognee example to see AI memory in action.

## Basic Usage

This minimal example shows how to add content, process it, and perform a search:

```python  theme={null}
import cognee
import asyncio

async def main():

    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # Add sample content
    text = "Cognee turns documents into AI memory."
    await cognee.add(text)
    
    # Process with LLMs to build the knowledge graph
    await cognee.cognify()
    
    # Search the knowledge graph
    results = await cognee.search(
        query_text="What does Cognee do?"
    )
    
    # Print
    for result in results:
        print(result)

if __name__ == '__main__':
    asyncio.run(main())
```

<Accordion title="Visualisation">
  <p>Interactive knowledge graph visualization -- drag nodes, zoom, and hover for details. Create your own visualization with 2 additional lines of code [here](/guides/graph-visualization).</p>

  <Frame width="100%" height="500px">
    <GraphVisualization />
  </Frame>
</Accordion>

## What just happened

The code demonstrates Cognee's three core operations:

* **`.add`** — Adds data to Cognee so they can be cognified. In this case, we added a single string ("Cognee turns documents into AI memory"); from Cognee's perspective, this string is a document.
* **`.cognify`** — This is where the cognification happens. All documents are chunked, entities are extracted, relationships are made, and summaries are generated. In this case, we can expect entities like Frodo, One Ring, and Mordor.
* **`.search`** — Queries the knowledge graph using vector similarity and graph traversal to find relevant information and return contextual results.

## About `async` / `await` in Cognee

<Note>
  **Cognee uses asynchronous code extensively.** That means many of its functions are defined with `async` and must be called with `await`. This lets Python handle waiting (e.g. for I/O or network calls) without blocking the rest of your program.
</Note>

<Accordion title="Async basics">
  This example uses <code>async</code> / <code>await</code>, Python’s way of doing asynchronous programming.
  Asynchronous programming is used when functions may block because they are waiting for something (for example, a reply from an API call). By writing <code>async def</code>, you define a function that can pause at certain points.
  The <code>await</code> keyword marks those calls that may need to pause.
  To run such functions, Python provides the <code>asyncio</code> library. It uses a loop, called the event loop, which executes your code in order but, whenever a function is waiting, can temporarily run another one. From inside your function, though, everything still runs top-to-bottom: each line after an <code>await</code> only executes once the awaited call has finished.
</Accordion>

<Accordion title="Async resources">
  * A good starting point is this [guide](https://realpython.com/async-io-python/).
  * Official documentation is available [here](https://docs.python.org/3/library/asyncio.html).
</Accordion>

## Next Steps

<Card title="Cognee core concepts" href="/core-concepts/overview" icon="compass">
  Learn about Cognee's core concepts, architecture, building blocks, and main operations.
</Card>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt