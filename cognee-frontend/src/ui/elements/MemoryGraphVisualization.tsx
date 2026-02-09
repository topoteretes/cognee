/**
 * Memory Graph Visualization
 *
 * Reusable visualization component with retrieval-first features:
 * - Node set inference and display
 * - Retrieval search with explanations
 * - Neutral-by-default highlighting
 * - Type attributes vs. inferred sets separation
 *
 * Works with any graph data (mock or real Cognee API data)
 */

"use client";

import { useState, useMemo } from "react";
import GraphVisualization from "@/ui/elements/GraphVisualization";
import { Edge, Node } from "@/ui/rendering/graph/types";
import { NodeSet } from "@/types/NodeSet";
import { inferNodeSets, mockRetrievalSearch } from "@/lib/inferNodeSets";
import type { RetrievalResult } from "@/types/NodeSet";

interface MemoryGraphVisualizationProps {
  nodes: Node[];
  edges: Edge[];
  title?: string;
  showControls?: boolean;
}

export default function MemoryGraphVisualization({
  nodes,
  edges,
  title = "Memory Retrieval Debugger",
  showControls = true,
}: MemoryGraphVisualizationProps) {
  const [showLegend, setShowLegend] = useState(true);

  // Retrieval-first: search replaces static filtering
  const [searchQuery, setSearchQuery] = useState("");
  const [retrievalResults, setRetrievalResults] = useState<RetrievalResult[]>([]);

  // Node sets: primary abstraction
  const [selectedNodeSet, setSelectedNodeSet] = useState<NodeSet | null>(null);

  // Layer visibility controls
  const [showNodes, setShowNodes] = useState(true);
  const [showEdges, setShowEdges] = useState(true);
  const [showMetaballs, setShowMetaballs] = useState(false);

  // Node Attributes section collapsed by default (secondary concern)
  const [showNodeAttributes, setShowNodeAttributes] = useState(false);

  // Infer node sets from graph structure (CRITICAL: separate attributes from sets)
  const { typeAttributes, inferredSets } = useMemo(() => {
    return inferNodeSets(nodes, edges, {
      minSetSize: 5,
      maxSets: 15,
    });
  }, [nodes, edges]);

  // Neutral-by-default: only highlight nodes that are selected or retrieved
  const highlightedNodeIds = useMemo(() => {
    const ids = new Set<string>();

    // Nodes from retrieval results
    retrievalResults.forEach(result => {
      if (result.type === "node" && result.nodeId) {
        ids.add(result.nodeId);
      } else if (result.type === "nodeSet" && result.nodeSet) {
        result.nodeSet.nodeIds.forEach(id => ids.add(id));
      }
    });

    // Nodes from selected set
    if (selectedNodeSet) {
      selectedNodeSet.nodeIds.forEach(id => ids.add(id));
    }

    return ids;
  }, [retrievalResults, selectedNodeSet]);

  // Handle retrieval search
  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.trim()) {
      const results = mockRetrievalSearch(query, nodes, inferredSets);
      setRetrievalResults(results);
    } else {
      setRetrievalResults([]);
    }
  };

  const handleReset = () => {
    setSearchQuery("");
    setRetrievalResults([]);
    setSelectedNodeSet(null);
  };

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-900 via-black to-purple-900 text-white">
      {/* Main Visualization */}
      <div className="flex-1 relative">
        <GraphVisualization
          nodes={nodes}
          edges={edges}
          config={{
            fontSize: 11,
            showNodes,
            showEdges,
            showMetaballs,
            highlightedNodeIds,
          }}
        />

        {/* Header */}
        <div className="absolute top-0 left-0 right-0 p-6 bg-gradient-to-b from-black/90 via-black/50 to-transparent pointer-events-none">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
            {title}
          </h1>
          <p className="text-gray-300">
            {highlightedNodeIds.size > 0 ? (
              <>
                <span className="text-purple-400 font-semibold">{highlightedNodeIds.size}</span> retrieved
                {" / "}
                <span className="text-gray-500">{nodes.length.toLocaleString()} total</span>
                {" • "}
                <span className="text-sm">{inferredSets.length} inferred sets</span>
              </>
            ) : (
              <>
                <span>{nodes.length.toLocaleString()} nodes</span>
                {" • "}
                <span>{inferredSets.length} inferred sets</span>
                {" • "}
                <span className="text-gray-500">Search to retrieve</span>
              </>
            )}
          </p>
        </div>

        {showControls && (
          <div className="absolute top-6 right-6 flex flex-col gap-3 pointer-events-auto max-w-md">
            {/* Retrieval Search */}
            <div className="relative">
              <input
                type="text"
                placeholder="🔍 Retrieve memories... (e.g., 'AI', 'Physics')"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="w-full px-4 py-2 bg-black/70 backdrop-blur-md border border-purple-500/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
              />
              {searchQuery && (
                <button
                  onClick={() => handleSearch("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  ✕
                </button>
              )}
            </div>

            {/* View Controls */}
            <div className="flex gap-2">
              <button
                onClick={() => setShowLegend(!showLegend)}
                className="flex-1 px-4 py-2 bg-black/70 hover:bg-black/90 backdrop-blur-md rounded-lg border border-purple-500/30 transition-all"
              >
                {showLegend ? "Hide" : "Show"} Panel
              </button>
              {(searchQuery || selectedNodeSet || retrievalResults.length > 0) && (
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-500 hover:to-orange-500 rounded-lg transition-all shadow-lg"
                >
                  Reset
                </button>
              )}
            </div>

            {/* Layer Visibility Controls */}
            <div className="bg-black/70 backdrop-blur-md rounded-lg p-3 border border-purple-500/30">
              <div className="text-xs font-semibold text-gray-400 mb-2">Layers</div>
              <div className="flex flex-col gap-2">
                <button
                  onClick={() => setShowNodes(!showNodes)}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg transition-all text-sm ${
                    showNodes
                      ? "bg-purple-600/30 border border-purple-500/50"
                      : "bg-white/5 border border-gray-600/30"
                  }`}
                >
                  <span>● Nodes</span>
                  <span className="text-xs">{showNodes ? "ON" : "OFF"}</span>
                </button>
                <button
                  onClick={() => setShowEdges(!showEdges)}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg transition-all text-sm ${
                    showEdges
                      ? "bg-amber-600/30 border border-amber-500/50"
                      : "bg-white/5 border border-gray-600/30"
                  }`}
                >
                  <span>─ Paths</span>
                  <span className="text-xs">{showEdges ? "ON" : "OFF"}</span>
                </button>
                <button
                  onClick={() => setShowMetaballs(!showMetaballs)}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg transition-all text-sm ${
                    showMetaballs
                      ? "bg-purple-600/20 border border-purple-500/30"
                      : "bg-white/5 border border-gray-600/30"
                  }`}
                >
                  <span>◉ Clouds</span>
                  <span className="text-xs">{showMetaballs ? "ON" : "OFF"}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Side Panel */}
      {showLegend && (
        <div className="w-96 bg-black/95 backdrop-blur-xl border-l border-purple-500/20 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Retrieval Results */}
            {retrievalResults.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Retrieved Memories
                </h2>
                <div className="space-y-3">
                  {retrievalResults.slice(0, 10).map((result, idx) => (
                    <div
                      key={idx}
                      className="bg-white/5 hover:bg-white/10 p-3 rounded-lg border border-purple-500/20 transition-all"
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="font-medium text-sm">
                          {result.type === "node" ? result.nodeLabel : result.nodeSet?.name}
                        </div>
                        <div className="text-xs px-2 py-0.5 bg-purple-600/30 rounded">
                          {(result.similarityScore * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-xs text-gray-400 mb-2">
                        {result.why}
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {result.signals.map((signal, sidx) => (
                          <span
                            key={sidx}
                            className="text-xs px-2 py-0.5 bg-black/30 rounded"
                          >
                            {signal.name}: {(signal.weight * 100).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Node Attributes - Secondary (NOT sets, just metadata) */}
            <div>
              <button
                onClick={() => setShowNodeAttributes(!showNodeAttributes)}
                className="w-full flex items-center justify-between mb-3 text-left"
              >
                <h3 className="text-sm font-semibold text-gray-400">
                  Node Attributes {showNodeAttributes ? "▼" : "▶"}
                </h3>
                <span className="text-xs text-gray-500">
                  {typeAttributes.size} types
                </span>
              </button>
              {showNodeAttributes && (
                <div className="space-y-1 mb-6 pl-2">
                  {Array.from(typeAttributes.entries())
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, count]) => (
                      <div
                        key={type}
                        className="flex items-center justify-between text-xs px-3 py-1.5 bg-white/5 rounded"
                      >
                        <span className="text-gray-400">type: {type}</span>
                        <span className="text-gray-500">({count})</span>
                      </div>
                    ))}
                </div>
              )}
            </div>

            {/* Inferred Node Sets - PRIMARY ABSTRACTION */}
            <div>
              <h2 className="text-xl font-bold mb-3 flex items-center justify-between">
                <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Node Sets
                </span>
                {selectedNodeSet && (
                  <button
                    onClick={() => setSelectedNodeSet(null)}
                    className="text-xs px-2 py-1 bg-red-600 hover:bg-red-500 rounded"
                  >
                    Clear
                  </button>
                )}
              </h2>
              <div className="space-y-2">
                {inferredSets.map((nodeSet) => {
                  const isSelected = selectedNodeSet?.id === nodeSet.id;
                  return (
                    <button
                      key={nodeSet.id}
                      onClick={() => setSelectedNodeSet(isSelected ? null : nodeSet)}
                      className={`w-full text-left p-3 rounded-lg transition-all ${
                        isSelected
                          ? "bg-gradient-to-r from-purple-600 to-pink-600 shadow-lg"
                          : "bg-white/5 hover:bg-white/10"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="font-medium text-sm">{nodeSet.name}</div>
                        <div className="text-xs px-2 py-0.5 bg-black/30 rounded">
                          {nodeSet.size}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Enhanced explanatory section with explicit semantics */}
            <div className="pt-6 border-t border-purple-500/20">
              <h3 className="font-semibold mb-3 text-purple-400">Visual Elements</h3>
              <div className="space-y-2 text-sm text-gray-400">
                <div className="flex items-start gap-2">
                  <span className="text-purple-400 mt-0.5 font-bold text-lg">●</span>
                  <div>
                    <strong className="text-gray-300">Node Size = Importance:</strong>
                    <div className="text-xs mt-0.5">Larger = Domain/Field (structural), Smaller = Application (leaf)</div>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">─</span>
                  <div>
                    <strong className="text-gray-300">Paths (zoom to see):</strong>
                    <div className="text-xs mt-0.5">Relationships • Hover node to highlight connections</div>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-purple-400/40 mt-0.5">◉</span>
                  <div>
                    <strong className="text-gray-300">Background Clouds:</strong>
                    <div className="text-xs mt-0.5">Conceptual Density • Visible at far zoom for cluster overview</div>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-cyan-400/60 mt-0.5">○</span>
                  <div>
                    <strong className="text-gray-300">Boundary Rings:</strong>
                    <div className="text-xs mt-0.5">Type Clusters • Spatial grouping by semantic category</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
