"use client";


import { useCallback, useRef, useState, MutableRefObject } from "react";
import { ForceGraphMethods } from "react-force-graph-2d";

import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import { Footer } from "@/ui/Partials";
import CrewAITrigger from "./CrewAITrigger";
import GraphVisualization from "./GraphVisualization";
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

  const graphControls = useRef<GraphControlsAPI>();

  const onActivityChange = (activities: any) => {
    graphControls.current?.updateActivity(activities);
  };

  const [graphShape, setGraphShape] = useState<string>("none");
  
  return (
    <main className="flex flex-col h-full">
      <div className="pt-6 pr-3 pb-3 pl-6">
        <TextLogo width={86} height={24} />
      </div>
      <Divider />
      <div className="w-full h-full relative overflow-hidden">
        {data && graphControls.current && (
          <GraphVisualization
            ref={graphRef as unknown as MutableRefObject<ForceGraphMethods>}
            data={data}
            graphShape={graphShape}
            graphControls={graphControls as unknown as MutableRefObject<GraphControlsAPI>}
          />
        )}

        <div className="absolute top-2 left-2 bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md max-w-2xl">
          <CogneeAddWidget onData={onDataChange} />
          <CrewAITrigger onData={onDataChange} onActivity={onActivityChange} />
        </div>

        <div className="absolute top-2 right-2 bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-110">
          <GraphControls
            ref={graphControls as unknown as MutableRefObject<GraphControlsAPI>}
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
