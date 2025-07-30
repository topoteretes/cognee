"use client";

import { MutableRefObject, useEffect, useImperativeHandle, useRef, useState, useCallback } from "react";
import { forceCollide, forceManyBody } from "d3-force-3d";
import ForceGraph, { ForceGraphMethods, GraphData, LinkObject, NodeObject } from "react-force-graph-2d";
import { GraphControlsAPI } from "./GraphControls";
import getColorForNodeType from "./getColorForNodeType";

interface GraphVisuzaliationProps {
  ref: MutableRefObject<GraphVisualizationAPI>;
  data?: GraphData<NodeObject, LinkObject>;
  graphControls: MutableRefObject<GraphControlsAPI>;
}

export interface GraphVisualizationAPI {
  zoomToFit: ForceGraphMethods["zoomToFit"];
  setGraphShape: (shape: string) => void;
}

export default function GraphVisualization({ ref, data, graphControls }: GraphVisuzaliationProps) {
  const textSize = 6;
  const nodeSize = 15;
  // const addNodeDistanceFromSourceNode = 15;

  // State for tracking container dimensions
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle resize
  const handleResize = useCallback(() => {
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current;
      setDimensions({ width: clientWidth, height: clientHeight });

      // Trigger graph refresh after resize
      if (graphRef.current) {
        // Small delay to ensure DOM has updated
        setTimeout(() => {
          graphRef.current?.zoomToFit(1000,50);
        }, 100);
      }
    }
  }, []);

  // Set up resize observer
  useEffect(() => {
    // Initial size calculation
    handleResize();

    // ResizeObserver 
    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [handleResize]);

  const handleNodeClick = (node: NodeObject) => {
    graphControls.current?.setSelectedNode(node);
    // ref.current?.d3ReheatSimulation()
  }

  const handleBackgroundClick = (/* event: MouseEvent */) => {
    const selectedNode = graphControls.current?.getSelectedNode();

    if (!selectedNode) {
      return;
    }

    graphControls.current?.setSelectedNode(null);

    // const graphBoundingBox = document.getElementById("graph-container")?.querySelector("canvas")?.getBoundingClientRect();
    // const x = event.clientX - graphBoundingBox!.x;
    // const y = event.clientY - graphBoundingBox!.y;

    // const graphClickCoords = graphRef.current!.screen2GraphCoords(x, y);

    // const distanceFromAddNode = Math.sqrt(
      // Math.pow(graphClickCoords.x - (selectedNode!.x! + addNodeDistanceFromSourceNode), 2)
    //   + Math.pow(graphClickCoords.y - (selectedNode!.y! + addNodeDistanceFromSourceNode), 2)
    // );

    // if (distanceFromAddNode <= 10) {
    //   enableAddNodeForm();
    // } else {
    //   disableAddNodeForm();
    //   graphControls.current?.setSelectedNode(null);
    // }
  };

  function renderNode(node: NodeObject, ctx: CanvasRenderingContext2D, globalScale: number, renderType: string = "replace") {
    // const selectedNode = graphControls.current?.getSelectedNode();

    ctx.save();

    // if (node.id === selectedNode?.id) {
    //   ctx.fillStyle = "gray";

    //   ctx.beginPath();
    //   ctx.arc(node.x! + addNodeDistanceFromSourceNode, node.y! + addNodeDistanceFromSourceNode, 10, 0, 2 * Math.PI);
    //   ctx.fill();

    //   ctx.beginPath();
    //   ctx.moveTo(node.x! + addNodeDistanceFromSourceNode - 5, node.y! + addNodeDistanceFromSourceNode)
    //   ctx.lineTo(node.x! + addNodeDistanceFromSourceNode - 5 + 10, node.y! + addNodeDistanceFromSourceNode);
    //   ctx.stroke();

    //   ctx.beginPath();
    //   ctx.moveTo(node.x! + addNodeDistanceFromSourceNode, node.y! + addNodeDistanceFromSourceNode - 5)
    //   ctx.lineTo(node.x! + addNodeDistanceFromSourceNode, node.y! + addNodeDistanceFromSourceNode - 5 + 10);
    //   ctx.stroke();
    // }

    if (renderType === "replace") {
      ctx.beginPath();
      ctx.fillStyle = getColorForNodeType(node.type);
      ctx.arc(node.x!, node.y!, nodeSize, 0, 2 * Math.PI);
      ctx.fill();
    }

    // draw text label (with background rect)
    const textPos = {
      x: node.x!,
      y: node.y!,
    };

    ctx.translate(textPos.x, textPos.y);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#333333";
    ctx.font = `${textSize}px Sans-Serif`;
    ctx.fillText(node.label, 0, 0);

    ctx.restore();
  }

  function renderLink(link: LinkObject, ctx: CanvasRenderingContext2D) {
    const MAX_FONT_SIZE = 4;
    const LABEL_NODE_MARGIN = nodeSize * 1.5;

    const start = link.source;
    const end = link.target;

    // ignore unbound links
    if (typeof start !== "object" || typeof end !== "object") return;

    const textPos = {
      x: start.x! + (end.x! - start.x!) / 2,
      y: start.y! + (end.y! - start.y!) / 2,
    };

    const relLink = { x: end.x! - start.x!, y: end.y! - start.y! };

    const maxTextLength = Math.sqrt(Math.pow(relLink.x, 2) + Math.pow(relLink.y, 2)) - LABEL_NODE_MARGIN * 2;

    let textAngle = Math.atan2(relLink.y, relLink.x);
    // maintain label vertical orientation for legibility
    if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle);
    if (textAngle < -Math.PI / 2) textAngle = -(-Math.PI - textAngle);

    const label = link.label

    // estimate fontSize to fit in link length
    ctx.font = "1px Sans-Serif";
    const fontSize = Math.min(MAX_FONT_SIZE, maxTextLength / ctx.measureText(label).width);
    ctx.font = `${fontSize}px Sans-Serif`;
    const textWidth = ctx.measureText(label).width;
    const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

    // draw text label (with background rect)
    ctx.save();
    ctx.translate(textPos.x, textPos.y);
    ctx.rotate(textAngle);

    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.fillRect(- bckgDimensions[0] / 2, - bckgDimensions[1] / 2, bckgDimensions[0], bckgDimensions[1]);

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "darkgrey";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }

  function renderInitialNode(node: NodeObject, ctx: CanvasRenderingContext2D, globalScale: number) {
    renderNode(node, ctx, globalScale, "after");
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  function handleDagError(loopNodeIds: (string | number)[]) {}

  const graphRef = useRef<ForceGraphMethods>();

  useEffect(() => {
    if (typeof window !== "undefined" && data && graphRef.current) {
      // add collision force
      graphRef.current.d3Force("collision", forceCollide(nodeSize * 1.5));
      graphRef.current.d3Force("charge", forceManyBody().strength(-1500).distanceMin(300).distanceMax(900));
    }
  }, [data, graphRef]);

  const [graphShape, setGraphShape] = useState<string>();
  
  useImperativeHandle(ref, () => ({
    zoomToFit: graphRef.current!.zoomToFit,
    setGraphShape: setGraphShape,
  }));

  return (
    <div ref={containerRef} className="w-full h-full" id="graph-container">
      {(data && typeof window !== "undefined") ? (
        <ForceGraph
          ref={graphRef}
          width={dimensions.width}
          height={dimensions.height}
          dagMode={graphShape as unknown as undefined}
          dagLevelDistance={300}
          onDagError={handleDagError}
          graphData={data}

          nodeLabel="label"
          nodeRelSize={nodeSize}
          nodeCanvasObject={renderNode}
          nodeCanvasObjectMode={() => "replace"}

          linkLabel="label"
          linkCanvasObject={renderLink}
          linkCanvasObjectMode={() => "after"}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}

          onNodeClick={handleNodeClick}
          onBackgroundClick={handleBackgroundClick}
          d3VelocityDecay={0.3}
        />
      ) : (
        <ForceGraph
          ref={graphRef}
          width={dimensions.width}
          height={dimensions.height}
          dagMode={graphShape as unknown as undefined}
          dagLevelDistance={100}
          graphData={{
            nodes: [{ id: 1, label: "Add" }, { id: 2, label: "Cognify" }, { id: 3, label: "Search" }],
            links: [{ source: 1, target: 2, label: "but don't forget to" }, { source: 2, target: 3, label: "and after that you can" }],
          }}

          nodeLabel="label"
          nodeRelSize={20}
          nodeCanvasObject={renderInitialNode}
          nodeCanvasObjectMode={() => "after"}
          nodeAutoColorBy="type"

          linkLabel="label"
          linkCanvasObject={renderLink}
          linkCanvasObjectMode={() => "after"}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
        />
      )}
    </div>
  );
}
