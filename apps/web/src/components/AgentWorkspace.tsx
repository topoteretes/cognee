"use client";

import React, { useState } from "react";
import { Play, RotateCcw, User, AlertCircle } from "lucide-react";

interface AgentProfile {
  name: string;
  role: string;
  avatar: string;
  color: string;
}

interface StepLog {
  step_index: number;
  agent: AgentProfile;
  action: "read" | "write";
  topic: string;
  content: string;
  log: string;
  recall_context?: string[];
  nodesUsed?: string[];
}

interface AgentWorkspaceProps {
  onAgentStepExecuted: (nodeIds: string[]) => void;
}

const AGENTS = [
  { id: "Architect", name: "Alex", role: "Architect", avatar: "📐", color: "#3B82F6", status: "Idle" },
  { id: "Developer", name: "Devin", role: "Developer", avatar: "💻", color: "#10B981", status: "Idle" },
  { id: "QA", name: "Quinn", role: "QA Engineer", avatar: "🔍", color: "#EF4444", status: "Idle" },
  { id: "Researcher", name: "Regina", role: "Researcher", avatar: "🔬", color: "#8B5CF6", status: "Idle" },
  { id: "TechWriter", name: "Wendy", role: "Technical Writer", avatar: "✍️", color: "#F59E0B", status: "Idle" }
];

export default function AgentWorkspace({ onAgentStepExecuted }: AgentWorkspaceProps) {
  const [logs, setLogs] = useState<StepLog[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [simulationFinished, setSimulationFinished] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Tracks active agent in this step
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null);

  const triggerNextStep = async () => {
    if (simulationFinished) return;
    setRunning(true);
    setError(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    try {
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/agents/sim/step", {
        method: "POST",
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      const data = await res.json();
      
      if (!res.ok) {
        if (res.status === 503 && data.status === "busy") {
          throw new Error("Service busy, please retry");
        }
        throw new Error(data.detail || data.message || "Step execution failed.");
      }

      if (data.status === "completed" && !data.step) {
        setSimulationFinished(true);
        setActiveAgentId(null);
        return;
      }

      const step = data.step as StepLog;
      setLogs(prev => [...prev, step]);
      setCurrentStep(data.step_index);
      
      // Update active agent visual state
      const matchingAgent = AGENTS.find(a => a.role === step.agent.role);
      if (matchingAgent) {
        setActiveAgentId(matchingAgent.id);
      }
      
      // Highlight nodes in the main graph
      if (step.nodesUsed && step.nodesUsed.length > 0) {
        onAgentStepExecuted(step.nodesUsed);
      } else if (step.content) {
        // Find matching nodes from graph to highlight
        const graphRes = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/graph/data");
        if (graphRes.ok) {
          const graphData = await graphRes.json();
          const nodes = graphData.nodes || [];
          const words = (step.content + " " + step.log).toLowerCase().split(/\s+/);
          const nodesUsed: string[] = [];
          for (const n of nodes) {
            const label = n.label.toLowerCase();
            if (words.some(w => w.length > 3 && (label.includes(w) || w.includes(label)))) {
              nodesUsed.push(n.id);
            }
          }
          onAgentStepExecuted(nodesUsed);
        }
      }

      if (data.status === "completed") {
        setSimulationFinished(true);
      }
    } catch (err: any) {
      setError(err.message || "Failed to run simulation step.");
    } finally {
      setRunning(false);
    }
  };

  const resetSim = async () => {
    setError(null);
    try {
      await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/agents/sim/reset", {
        method: "POST"
      });
      setLogs([]);
      setCurrentStep(0);
      setSimulationFinished(false);
      setActiveAgentId(null);
      onAgentStepExecuted([]);
    } catch (err: any) {
      setError(err.message || "Failed to reset simulation.");
    }
  };

  const getAgentStatus = (agentId: string) => {
    if (activeAgentId === agentId) {
      const activeStep = SIMULATION_STEPS_LOOKUP[currentStep];
      if (activeStep) {
        return activeStep.action === "write" ? "Writing Memory" : "Recalling Memory";
      }
      return "Active";
    }
    return "Idle";
  };

  // Mock mapping of steps for status labels
  const SIMULATION_STEPS_LOOKUP = [
    { action: "write" }, { action: "read" }, { action: "write" },
    { action: "read" }, { action: "write" }, { action: "read" },
    { action: "write" }, { action: "read" }, { action: "write" },
    { action: "read" }, { action: "write" }
  ];

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      <div className="flex justify-between items-start mb-1">
        <div>
          <h2 className="text-lg font-bold text-white font-outfit">Multi-Agent Memory Workspace</h2>
          <p className="text-slate-500 text-xs">
            Watch cooperative AI agents share knowledge, log bugs, and fix codes on a unified Cognee graph.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={resetSim}
            className="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5 font-medium"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
          <button
            onClick={triggerNextStep}
            disabled={running || simulationFinished}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white text-xs px-4 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 shadow shadow-blue-500/25 animate-pulse"
          >
            <Play className="w-3.5 h-3.5" />
            {simulationFinished ? "Completed" : running ? "Running..." : "Play Next Step"}
          </button>
        </div>
      </div>

      {/* Agents Row */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-4 mb-5">
        {AGENTS.map((a) => {
          const status = getAgentStatus(a.id);
          const isActive = activeAgentId === a.id;
          return (
            <div
              key={a.id}
              className={`p-3 rounded-xl border flex flex-col items-center text-center transition-all ${
                isActive
                  ? "bg-slate-950/40 border-blue-500/50 shadow shadow-blue-500/10 scale-105"
                  : "bg-slate-950/20 border-slate-800"
              }`}
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-xl mb-2 relative`}
                style={{ backgroundColor: `${a.color}20`, border: `2.5px solid ${isActive ? "#3B82F6" : a.color}` }}
              >
                {a.avatar}
                {isActive && (
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 border-2 border-slate-950 rounded-full animate-ping"></span>
                )}
              </div>
              <div className="text-xs font-bold text-white leading-none">{a.name}</div>
              <div className="text-[9px] text-slate-500 font-medium mt-1 leading-none">{a.role}</div>
              
              <span className={`mt-2 px-2 py-0.5 rounded text-[8px] font-mono leading-none border ${
                status === "Idle"
                  ? "bg-slate-950 border-slate-900 text-slate-600"
                  : status.includes("Writing")
                    ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400 font-bold"
                    : "bg-purple-500/10 border-purple-500/20 text-purple-400 font-bold animate-pulse"
              }`}>
                {status}
              </span>
            </div>
          );
        })}
      </div>

      {/* Activity Feed */}
      <div className="flex-1 border border-slate-800/80 rounded-xl p-4 bg-slate-950/30 overflow-y-auto h-full flex flex-col justify-between min-h-[180px]">
        <div className="space-y-3 flex-1 overflow-y-auto pr-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono border-b border-slate-900 pb-2 mb-2 flex items-center justify-between">
            <span>Agent Transaction Logs</span>
            <span>Step {currentStep} / 11</span>
          </div>

          {logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-slate-500 h-full">
              <User className="w-10 h-10 text-slate-800 mb-2" />
              <span className="text-xs font-semibold text-slate-400">Collaboration Space Ready</span>
              <span className="text-[10px] text-slate-600 mt-0.5">Click "Play Next Step" to start the simulation.</span>
            </div>
          ) : (
            logs.map((log, idx) => {
              const isWrite = log.action === "write";
              return (
                <div key={idx} className="flex gap-3 bg-slate-950/40 border border-slate-900 p-3.5 rounded-xl hover:border-slate-850 transition">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 border text-sm"
                    style={{ backgroundColor: `${log.agent.color}15`, borderColor: log.agent.color }}
                  >
                    {log.agent.avatar}
                  </div>
                  <div className="flex-1 space-y-1.5 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white">
                        {log.agent.name} <span className="text-[10px] text-slate-500 font-normal">({log.agent.role})</span>
                      </span>
                      <span className={`px-2 py-0.5 rounded text-[8px] font-mono leading-none border ${
                        isWrite
                          ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                          : "bg-purple-500/10 border-purple-500/20 text-purple-400"
                      }`}>
                        {isWrite ? "remember()" : "recall()"}
                      </span>
                    </div>

                    <div className="text-xs font-semibold text-slate-300 font-sans leading-relaxed">
                      {log.log}
                    </div>

                    {isWrite ? (
                      <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-900 text-[10px] leading-relaxed text-slate-400 font-mono">
                        <span className="text-emerald-500 font-bold block mb-1">
                          [Remembered Topic: {log.topic}]
                        </span>
                        {log.content}
                      </div>
                    ) : (
                      <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-900 text-[10px] leading-relaxed text-slate-400 font-mono space-y-1">
                        <span className="text-purple-400 font-bold block">
                          [Queried context: "{log.content}"]
                        </span>
                        {log.recall_context && log.recall_context.length > 0 ? (
                          log.recall_context.map((c, cidx) => (
                            <div key={cidx} className="pl-3 border-l border-slate-800 text-slate-500 text-[9px] truncate">
                              • {c}
                            </div>
                          ))
                        ) : (
                          <div className="text-slate-600 italic">No contexts returned (blank slate search).</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
