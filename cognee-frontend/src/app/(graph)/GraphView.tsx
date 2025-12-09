"use client";

import { useCallback, useRef, useState, RefObject } from "react";

import Link from "next/link";
import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import { Footer } from "@/ui/Partials";
import GraphLegend from "./GraphLegend";
import { DiscordIcon, GithubIcon } from "@/ui/Icons";
import ActivityLog, { ActivityLogAPI } from "./ActivityLog";
import GraphControls, { GraphControlsAPI } from "./GraphControls";
import CogneeAddWidget, { NodesAndLinks } from "./CogneeAddWidget";
import GraphVisualization, { GraphVisualizationAPI } from "./GraphVisualization";

import { useBoolean } from "@/utils";

interface GraphNode {
  id: string | number;
  label: string;
  properties?: object;
}

interface GraphData {
  nodes: GraphNode[];
  links: { source: string | number; target: string | number; label: string }[];
}

export default function GraphView() {
  const {
    value: isAddNodeFormOpen,
  } = useBoolean(false);

  const [data, updateData] = useState<GraphData>();

  const onDataChange = useCallback((newData: NodesAndLinks) => {
    if (newData === null) {
      // Requests for resetting the data
      updateData(undefined);
      return;
    }

    if (!newData.nodes.length && !newData.links.length) {
      return;
    }

    updateData(newData);
  }, []);

  const graphRef = useRef<GraphVisualizationAPI>(null);

  const graphControls = useRef<GraphControlsAPI>(null);

  const activityLog = useRef<ActivityLogAPI>(null);

  return (
    <main className="flex flex-col h-full">
      <div className="flex flex-row justify-between items-center pt-6 pr-6 pb-6 pl-6">
        <TextLogo width={86} height={24} />

        <span className="flex flex-row items-center gap-8">
          <Link target="_blank" href="https://www.cognee.ai/">
            <span>Cognee Home</span>
          </Link>
          <Link target="_blank" href="https://github.com/topoteretes/cognee">
            <GithubIcon color="black" />
          </Link>
          <Link target="_blank" href="https://discord.gg/m63hxKsp4p">
            <DiscordIcon color="black" />
          </Link>
        </span>
      </div>
      <Divider />
      <div className="w-full h-full relative overflow-hidden">
        <GraphVisualization
          key={data?.nodes.length}
          ref={graphRef as RefObject<GraphVisualizationAPI>}
          data={data}
          graphControls={graphControls as RefObject<GraphControlsAPI>}
        />

        <div className="absolute top-2 left-2 flex flex-col gap-2">
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-sm">
            <CogneeAddWidget onData={onDataChange} />
          </div>
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-sm">
            <h2 className="text-xl text-white mb-4">Activity Log</h2>
            <ActivityLog ref={activityLog as RefObject<ActivityLogAPI>} />
          </div>
        </div>

        <div className="absolute top-2 right-2 flex flex-col gap-2 items-end">
          <div className="bg-gray-500 pt-4 pr-4 pb-4 pl-4 rounded-md w-110">
            <GraphControls
              data={data}
              ref={graphControls as RefObject<GraphControlsAPI>}
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
