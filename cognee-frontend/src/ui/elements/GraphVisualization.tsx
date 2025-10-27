"use client";

import classNames from 'classnames';
import { useEffect, useRef } from "react";

import { Edge, Node } from "@/ui/rendering/graph/types";
import animate from "@/ui/rendering/animate";

interface GraphVisualizationProps {
  nodes: Node[];
  edges: Edge[];
  className?: string;
  config?: { fontSize: number };
}

export default function GraphVisualization({
  nodes,
  edges,
  className,
  config,
}: GraphVisualizationProps) {
  const visualizationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const visualizationContainer = visualizationRef.current;

    if (visualizationContainer) {
      animate(nodes, edges, visualizationContainer, config);
    }
  }, [config, edges, nodes]);

  return (
    <div className={classNames("min-w-full min-h-full", className)} ref={visualizationRef} />
  );
}
