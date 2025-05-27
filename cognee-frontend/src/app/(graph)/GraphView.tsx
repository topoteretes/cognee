"use client";

import { forceCollide, forceManyBody } from "d3-force-3d";
import { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph, { ForceGraphMethods, LinkObject, NodeObject } from "react-force-graph-2d";

import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import { Footer } from "@/ui/Partials";
import CrewAITrigger from "./CrewAITrigger";
import CogneeAddWidget, { NodesAndEdges } from "./CogneeAddWidget";
import GraphControls, { GraphControlsAPI } from "./GraphControls";

import { useBoolean } from "@/utils";

// import exampleData from "./example_data.json";

interface GraphNode {
  id: string | number;
  label: string;
  properties?: {};
}

interface GraphData {
  nodes: GraphNode[];
  links: { source: string | number; target: string | number; label: string }[];
}

export default function GraphView() {
  const {
    value: isAddNodeFormOpen,
    setTrue: enableAddNodeForm,
    setFalse: disableAddNodeForm,
  } = useBoolean(false);

  const [data, updateData] = useState<GraphData | null>(null);

  const onDataChange = useCallback((newData: NodesAndEdges) => {
    if (!newData.nodes.length && !newData.links.length) {
      return;
    }

    updateData({
      nodes: newData.nodes,
      links: newData.links,
    });
  }, []);

  const graphRef = useRef<ForceGraphMethods>();

  const graphControls = useRef<GraphControlsAPI>(null);

  const onActivityChange = (activities: any) => {
    graphControls.current?.updateActivity(activities);
  };

  const handleNodeClick = (node: NodeObject) => {
    graphControls.current?.setSelectedNode(node);
    graphRef.current?.d3ReheatSimulation();
  };

  const textSize = 6;
  const nodeSize = 15;
  const addNodeDistanceFromSourceNode = 15;

  const handleBackgroundClick = (event: MouseEvent) => {
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
    //   Math.pow(graphClickCoords.x - (selectedNode!.x! + addNodeDistanceFromSourceNode), 2)
    //   + Math.pow(graphClickCoords.y - (selectedNode!.y! + addNodeDistanceFromSourceNode), 2)
    // );

    // if (distanceFromAddNode <= 10) {
    //   enableAddNodeForm();
    // } else {
    //   disableAddNodeForm();
    //   graphControls.current?.setSelectedNode(null);
    // }
  };

  function renderNode(node: NodeObject, ctx: CanvasRenderingContext2D, globalScale: number) {
    const selectedNode = graphControls.current?.getSelectedNode();

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

    // ctx.beginPath();
    // ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
    // ctx.fill();

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

  function handleDagError(loopNodeIds: (string | number)[]) {}

  useEffect(() => {
    // add collision force
    graphRef.current!.d3Force("collision", forceCollide(nodeSize * 1.5));
    graphRef.current!.d3Force("charge", forceManyBody().strength(-1500).distanceMin(300).distanceMax(900));
  }, [data]);

  const [graphShape, setGraphShape] = useState<string | undefined>(undefined);
  
  return (
    <main className="flex flex-col h-full">
      <div className="pt-6 pr-3 pb-3 pl-6">
        <TextLogo width={86} height={24} />
      </div>
      <Divider />
      <div className="w-full h-full relative overflow-hidden">
        <div className="w-full h-full" id="graph-container">
          {data ? (
            <ForceGraph
              ref={graphRef}
              dagMode={graphShape as undefined}
              dagLevelDistance={300}
              onDagError={handleDagError}
              graphData={data}

              nodeLabel="label"
              nodeRelSize={nodeSize}
              nodeCanvasObject={renderNode}
              nodeCanvasObjectMode={() => "after"}
              nodeAutoColorBy="type"

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
              dagMode="lr"
              dagLevelDistance={100}
              graphData={{
                nodes: [{ id: 1, label: "Add" }, { id: 2, label: "Cognify" }, { id: 3, label: "Search" }],
                links: [{ source: 1, target: 2, label: "but don't forget to" }, { source: 2, target: 3, label: "and after that you can" }],
              }}

              nodeLabel="label"
              nodeRelSize={20}
              nodeCanvasObject={renderNode}
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

        <div className="absolute top-2 left-2 bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md max-w-2xl">
          <CogneeAddWidget onData={onDataChange} />
          <CrewAITrigger onData={onDataChange} onActivity={onActivityChange} />
        </div>

        <div className="absolute top-2 right-2 bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-110">
          <GraphControls
            ref={graphControls}
            isAddNodeFormOpen={isAddNodeFormOpen}
            onFitIntoView={() => graphRef.current?.zoomToFit(1000, 50)}
            onGraphShapeChange={setGraphShape}
          />
        </div>
      </div>
      <Divider />
      <div className="pl-6 pr-6">
        <Footer>
          {(data?.nodes.length || data?.links.length) && (
            <div className="flex flex-row items-center gap-6">
              <span>Nodes: {data?.nodes.length || 0}</span>
              <span>Edges: {data?.links.length || 0}</span>
            </div>
          )}
        </Footer>
      </div>
    </main>
  );
}
