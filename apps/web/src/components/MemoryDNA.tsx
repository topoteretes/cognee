"use client";

import React from "react";
import { Activity, ShieldCheck, Zap, Hash, Clock, Link2, FileText, CornerDownRight } from "lucide-react";

interface Node {
  id: string;
  label: string;
  type: string;
  properties: any;
  heat: number;
  dna: {
    importance: number;
    freshness: number;
    trust: number;
    frequency: number;
    age_seconds: number;
    connections: number;
  };
}

interface MemoryDNAProps {
  node: Node | null;
  onClearSelection: () => void;
}

export default function MemoryDNA({ node, onClearSelection }: MemoryDNAProps) {
  if (!node) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-slate-900/40 backdrop-blur-md border border-slate-800 rounded-2xl">
        <Activity className="w-12 h-12 text-slate-700 animate-pulse mb-3" />
        <h3 className="text-slate-400 font-bold mb-1 font-outfit text-sm">No Memory Selected</h3>
        <p className="text-slate-500 text-xs max-w-xs">
          Click on any node in the Memory Galaxy to inspect its DNA profile, provenance, and extraction metadata.
        </p>
      </div>
    );
  }

  // Format age
  const formatAge = (seconds: number) => {
    if (seconds < 60) return `${seconds}s ago`;
    const mins = Math.floor(seconds / 60);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ${mins % 60}m ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ${hrs % 24}h ago`;
  };

  const dna = node.dna;

  return (
    <div className="w-full h-full flex flex-col bg-slate-900/60 backdrop-blur-lg border border-slate-800 rounded-2xl overflow-hidden shadow-xl text-slate-200">
      {/* Title */}
      <div className="p-4 border-b border-slate-800 flex justify-between items-start bg-slate-950/40">
        <div>
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {node.type}
            </span>
          </div>
          <h2 className="text-md font-bold text-white font-outfit truncate max-w-[200px]" title={node.label}>
            {node.label}
          </h2>
        </div>
        <button 
          onClick={onClearSelection}
          className="text-xs text-slate-500 hover:text-slate-300 font-medium px-2 py-1 rounded hover:bg-slate-800 transition"
        >
          Close
        </button>
      </div>

      {/* DNA Helix Metrics */}
      <div className="p-4 flex-1 overflow-y-auto space-y-4">
        <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">
          Memory Synapse DNA
        </div>

        <div className="grid grid-cols-1 gap-3">
          {/* Importance */}
          <div className="p-3 bg-slate-950/30 border border-slate-800/80 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-yellow-400" />
                Importance (Centrality)
              </span>
              <span className="font-bold text-yellow-400">{Math.round(dna.importance * 100)}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-yellow-500 to-amber-400 rounded-full transition-all duration-500" 
                style={{ width: `${dna.importance * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Freshness */}
          <div className="p-3 bg-slate-950/30 border border-slate-800/80 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-emerald-400" />
                Freshness (Decay Rate)
              </span>
              <span className="font-bold text-emerald-400">{Math.round(dna.freshness * 100)}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-500 to-green-400 rounded-full transition-all duration-500" 
                style={{ width: `${dna.freshness * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Trust */}
          <div className="p-3 bg-slate-950/30 border border-slate-800/80 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                Extraction Trust Score
              </span>
              <span className="font-bold text-cyan-400">{Math.round(dna.trust * 100)}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-400 rounded-full transition-all duration-500" 
                style={{ width: `${dna.trust * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Temporal / Network Stats */}
        <div className="grid grid-cols-2 gap-3 pt-1">
          <div className="p-2.5 bg-slate-950/40 border border-slate-800/60 rounded-xl flex items-center gap-2">
            <Hash className="w-4 h-4 text-purple-400" />
            <div>
              <div className="text-[10px] text-slate-500 uppercase font-mono">Recalls</div>
              <div className="text-xs font-bold text-slate-200">{dna.frequency} hits</div>
            </div>
          </div>

          <div className="p-2.5 bg-slate-950/40 border border-slate-800/60 rounded-xl flex items-center gap-2">
            <Link2 className="w-4 h-4 text-pink-400" />
            <div>
              <div className="text-[10px] text-slate-500 uppercase font-mono">Degrees</div>
              <div className="text-xs font-bold text-slate-200">{dna.connections} links</div>
            </div>
          </div>

          <div className="p-2.5 bg-slate-950/40 border border-slate-800/60 rounded-xl flex items-center gap-2 col-span-2">
            <Clock className="w-4 h-4 text-amber-500" />
            <div>
              <div className="text-[10px] text-slate-500 uppercase font-mono">Synapse Age</div>
              <div className="text-xs font-bold text-slate-200">{formatAge(dna.age_seconds)}</div>
            </div>
          </div>
        </div>

        {/* Extraction Properties */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono flex items-center gap-1">
            <FileText className="w-3.5 h-3.5 text-blue-400" />
            Extracted Properties
          </div>
          <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-3 max-h-[180px] overflow-y-auto font-mono text-[10px] space-y-2 shadow-inner">
            {Object.entries(node.properties).length === 0 ? (
              <span className="text-slate-600 italic">No extra properties.</span>
            ) : (
              Object.entries(node.properties).map(([key, val]) => {
                if (["vx", "vy", "x", "y"].includes(key)) return null;
                return (
                  <div key={key} className="border-b border-slate-900 pb-1.5 last:border-0 last:pb-0">
                    <span className="text-blue-400 block font-bold">{key}</span>
                    <span className="text-slate-300 block break-all font-sans text-[11px] leading-relaxed">
                      {typeof val === "object" ? JSON.stringify(val) : String(val)}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
