"use client";

import { useState, useMemo } from "react";
import { generateOntologyGraph } from "@/lib/generateOntologyGraph";
import MemoryGraphVisualization from "@/ui/elements/MemoryGraphVisualization";

type GraphMode = "small" | "medium" | "large";

export default function VisualizationDemoPage() {
  const [graphMode, setGraphMode] = useState<GraphMode>("medium");
  const [isGenerating, setIsGenerating] = useState(false);

  // Generate graph based on mode
  const { nodes, edges } = useMemo(() => {
    console.log(`Generating ${graphMode} ontology graph...`);
    setIsGenerating(true);

    let result;
    switch (graphMode) {
      case "small":
        result = { ...generateOntologyGraph("simple"), clusters: new Map() };
        break;
      case "medium":
        result = generateOntologyGraph("medium");
        break;
      case "large":
        result = generateOntologyGraph("complex");
        break;
    }

    setTimeout(() => setIsGenerating(false), 500);
    return result;
  }, [graphMode]);

  return (
    <div className="relative min-h-screen">
      {isGenerating ? (
        <div className="absolute inset-0 flex items-center justify-center bg-black/90 z-50 backdrop-blur-sm">
          <div className="text-center">
            <div className="relative">
              <div className="text-6xl mb-4 animate-spin">⚛️</div>
              <div className="absolute inset-0 text-6xl mb-4 animate-ping opacity-20">⚛️</div>
            </div>
            <div className="text-2xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Building Knowledge Graph...
            </div>
            <div className="text-gray-400">
              Creating {
                graphMode === "small" ? "~500" :
                graphMode === "medium" ? "~1,000" :
                "~1,500"
              } interconnected nodes
            </div>
          </div>
        </div>
      ) : null}

      {/* Mode Selector Overlay */}
      <div className="absolute top-6 left-6 z-10 pointer-events-auto">
        <div className="flex gap-1 bg-black/70 backdrop-blur-md rounded-lg p-1 border border-purple-500/30">
          {(["small", "medium", "large"] as GraphMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setGraphMode(mode)}
              className={`flex-1 px-3 py-2 rounded transition-all text-sm font-medium ${
                graphMode === mode
                  ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg shadow-purple-500/50"
                  : "hover:bg-white/10 text-gray-300"
              }`}
            >
              {mode === "small" && "500"}
              {mode === "medium" && "1K"}
              {mode === "large" && "1.5K"}
            </button>
          ))}
        </div>
      </div>

      <MemoryGraphVisualization
        nodes={nodes}
        edges={edges}
        title="Memory Retrieval Debugger (Demo)"
        showControls={true}
      />
    </div>
  );
}
