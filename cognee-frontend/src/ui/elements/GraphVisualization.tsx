"use client";

import classNames from 'classnames';
import { useEffect, useRef } from "react";

import { Edge, Node } from "@/ui/rendering/graph/types";
import animate from "@/ui/rendering/animate";

// IMPROVEMENT #8: Extended config for layered view controls
interface GraphVisualizationProps {
  nodes: Node[];
  edges: Edge[];
  className?: string;
  config?: {
    fontSize?: number;
    showNodes?: boolean;    // Toggle node visibility
    showEdges?: boolean;    // Toggle edge/path visibility
    showMetaballs?: boolean; // Toggle density cloud visibility
    highlightedNodeIds?: Set<string>; // Nodes to highlight (neutral-by-default)
  };
}

export default function GraphVisualization({
  nodes,
  edges,
  className,
  config,
}: GraphVisualizationProps) {
  const visualizationRef = useRef<HTMLDivElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const visualizationContainer = visualizationRef.current;

    if (visualizationContainer) {
      // Clean up previous visualization
      if (cleanupRef.current) {
        cleanupRef.current();
      }

      // Clear the container
      while (visualizationContainer.firstChild) {
        visualizationContainer.removeChild(visualizationContainer.firstChild);
      }

      // Create new visualization
      const cleanup = animate(nodes, edges, visualizationContainer, config);
      cleanupRef.current = cleanup;
    }

    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, [config, edges, nodes]);

  return (
    <div className={classNames("min-w-full min-h-full", className)} ref={visualizationRef} />
  );
}
