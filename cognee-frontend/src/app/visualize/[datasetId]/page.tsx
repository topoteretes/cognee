"use client";

import { useEffect, useState } from "react";
import { fetch } from "@/utils";
import { adaptCogneeGraphData, validateCogneeGraphResponse } from "@/lib/adaptCogneeGraphData";
import { CogneeGraphResponse } from "@/types/CogneeAPI";
import MemoryGraphVisualization from "@/ui/elements/MemoryGraphVisualization";
import { Edge, Node } from "@/ui/rendering/graph/types";

interface VisualizePageProps {
  params: { datasetId: string };
}

export default function Page({ params }: VisualizePageProps) {
  const [graphData, setGraphData] = useState<{ nodes: Node[], edges: Edge[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function getData() {
      try {
        setLoading(true);
        setError(null);

        const datasetId = (await params).datasetId;
        const response = await fetch(`/v1/datasets/${datasetId}/graph`);

        if (!response.ok) {
          throw new Error(`Failed to fetch graph data: ${response.statusText}`);
        }

        const apiData = await response.json();

        // Validate API response
        if (!validateCogneeGraphResponse(apiData)) {
          throw new Error("Invalid graph data format from API");
        }

        // Adapt Cognee API format to visualization format
        const adaptedData = adaptCogneeGraphData(apiData as CogneeGraphResponse);
        setGraphData(adaptedData);
      } catch (err) {
        console.error("Error loading graph data:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }
    getData();
  }, [params]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-900 via-black to-purple-900">
        <div className="text-center">
          <div className="text-6xl mb-4 animate-spin">⚛️</div>
          <div className="text-2xl font-bold text-white">Loading graph data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-900 via-black to-purple-900">
        <div className="text-center max-w-md p-6">
          <div className="text-6xl mb-4">⚠️</div>
          <div className="text-2xl font-bold text-white mb-2">Error Loading Graph</div>
          <div className="text-gray-400">{error}</div>
        </div>
      </div>
    );
  }

  if (!graphData || graphData.nodes.length === 0) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-900 via-black to-purple-900">
        <div className="text-center max-w-md p-6">
          <div className="text-6xl mb-4">📊</div>
          <div className="text-2xl font-bold text-white mb-2">No Graph Data</div>
          <div className="text-gray-400">This dataset has no graph data to visualize.</div>
        </div>
      </div>
    );
  }

  return (
    <MemoryGraphVisualization
      nodes={graphData.nodes}
      edges={graphData.edges}
      title="Cognee Memory Graph"
      showControls={true}
    />
  );
}
