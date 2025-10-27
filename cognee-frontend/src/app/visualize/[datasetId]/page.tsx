"use client";

import { useEffect, useState } from "react";
import { fetch } from "@/utils";

import { Edge, Node } from "@/ui/rendering/graph/types";
import GraphVisualization from "@/ui/elements/GraphVisualization";

interface VisualizePageProps {
  params: { datasetId: string };
}

export default function Page({ params }: VisualizePageProps) {
  const [graphData, setGraphData] = useState<{ nodes: Node[], edges: Edge[] }>();
  useEffect(() => {
    async function getData() {
      const datasetId = (await params).datasetId;
      const response = await fetch(`/v1/datasets/${datasetId}/graph`);
      const newGraphData = await response.json();
      setGraphData(newGraphData);
    }
    getData();
  }, [params]);

  return (
    <div className="flex min-h-screen">
      {graphData && (
        <GraphVisualization
          nodes={graphData.nodes}
          edges={graphData.edges}
          config={{
            fontSize: 10,
          }}
        />
      )}
    </div>
  );
}
