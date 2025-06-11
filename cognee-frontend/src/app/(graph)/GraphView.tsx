"use client";

import Image from "next/image";
import { useCallback, useRef, useState, MutableRefObject } from "react";

import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import { Footer } from "@/ui/Partials";
import GraphVisualization, { GraphVisualizationAPI } from "./GraphVisualization";
import CogneeAddWidget, { NodesAndEdges } from "./CogneeAddWidget";
import GraphControls, { GraphControlsAPI } from "./GraphControls";
import GraphLegend from "./GraphLegend";
import ActivityLog, { ActivityLogAPI } from "./ActivityLog";

import { useBoolean } from "@/utils";

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

  const [data, updateData] = useState<GraphData>();

  const onDataChange = useCallback((newData: NodesAndEdges) => {
    if (newData === null) {
      // Requests for resetting the data
      updateData(undefined);
      return;
    }

    if (!newData.nodes.length && !newData.links.length) {
      return;
    }

    updateData({
      nodes: newData.nodes,
      links: newData.links,
    });
  }, []);

  const graphRef = useRef<GraphVisualizationAPI>();

  const graphControls = useRef<GraphControlsAPI>();

  const activityLog = useRef<ActivityLogAPI>();

  return (
    <main className="flex flex-col h-full">
      <div className="flex flex-row justify-between items-center pt-6 pr-3 pb-3 pl-6">
        <TextLogo width={86} height={24} />

        <div className="flex flex-row items-center gap-8">
          Powered by{" "}

          <span className="flex flex-row gap-8">
            <Image src="/images/neo4j.png" alt="Neo4j" width={132} height={48} />
            <Image src="/images/lancedb.svg" alt="LanceDB" width={132} height={48} />
            <Image src="/images/deepnote.svg" alt="DeepNote" width={132} height={48} />
            <Image src="/images/crewai.png" alt="CrewAI" width={132} height={48} />
          </span>
        </div>
      </div>
      <Divider />
      <div className="w-full h-full relative overflow-hidden">
        <GraphVisualization
          key={data?.nodes.length}
          ref={graphRef as MutableRefObject<GraphVisualizationAPI>}
          data={data}
          graphControls={graphControls as MutableRefObject<GraphControlsAPI>}
        />

        <div className="absolute top-2 left-2 flex flex-col gap-2">
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-xs">
            <CogneeAddWidget onData={onDataChange} />
          </div>
          {/* <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-xs">
            <CrewAITrigger onData={onDataChange} onActivity={(activities) => activityLog.current?.updateActivityLog(activities)} />
          </div> */}
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-xs">
            <h2 className="text-xl text-white mb-4">Activity Log</h2>
            <ActivityLog ref={activityLog as MutableRefObject<ActivityLogAPI>} />
          </div>
        </div>

        <div className="absolute top-2 right-2 flex flex-col gap-2 items-end">
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-110">
            <GraphControls
              data={data}
              ref={graphControls as MutableRefObject<GraphControlsAPI>}
              isAddNodeFormOpen={isAddNodeFormOpen}
              onFitIntoView={() => graphRef.current!.zoomToFit(1000, 50)}
              onGraphShapeChange={(shape) => graphRef.current!.setGraphShape(shape)}
            />
          </div>
          {data?.nodes.length && (
            <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-48">
              <GraphLegend data={data?.nodes} />
            </div>
          )}
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
