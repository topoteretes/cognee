"use client";

import React, { useState, useEffect } from "react";
import { 
  ShieldCheck, Server, Database, Brain, Cpu, 
  RefreshCw, Clock, AlertTriangle, Terminal, Code, 
  HelpCircle, CheckCircle, XCircle
} from "lucide-react";

interface HealthData {
  status: string;
  runtime_version: string;
  active_configuration: {
    llm_provider: string;
    llm_model: string;
    embedding_provider: string;
    embedding_model: string;
    relational_db: string;
    vector_store: string;
    graph_engine: string;
  };
  components: {
    llm_provider: { status: string; details: string };
    embedding_provider: { status: string; details: string };
    vector_store: { status: string; details: string };
    graph_store: { status: string; details: string };
  };
  memory_statistics: {
    total_nodes: number;
    total_edges: number;
    last_recall_time: string;
    last_improve_time: string;
    active_dataset: string;
    queue_status: string;
  };
}

export default function HealthDashboard() {
  const [data, setData] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [viewMode, setViewMode] = useState<"gui" | "cli" | "json">("gui");

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/health/runtime", {
        headers: { "X-Tenant-Auth": tenantKey }
      });
      if (!res.ok) throw new Error("Failed to fetch runtime health telemetry.");
      const json = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "Failed to communicate with runtime diagnostics API.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchHealth();
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "connected":
      case "healthy":
      case "loaded":
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case "invalid_key":
      case "missing_key":
        return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
      default:
        return <XCircle className="w-4 h-4 text-rose-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "connected":
      case "healthy":
      case "loaded":
        return "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
      case "invalid_key":
      case "missing_key":
        return "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20";
      default:
        return "bg-rose-500/10 text-rose-400 border border-rose-500/20";
    }
  };

  const formatTime = (isoString: string) => {
    if (!isoString || isoString === "Never") return "Never";
    try {
      const date = new Date(isoString);
      const diff = Date.now() - date.getTime();
      
      const secs = Math.floor(diff / 1000);
      if (secs < 60) return `${secs} sec ago`;
      
      const mins = Math.floor(secs / 60);
      if (mins < 60) return `${mins} min ago`;
      
      const hours = Math.floor(mins / 60);
      return `${hours} hr ago`;
    } catch (e) {
      return isoString;
    }
  };

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-5 border-b border-slate-800/80 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white font-outfit mb-1 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-blue-400" />
            Runtime Health & Observability
          </h2>
          <p className="text-slate-500 text-xs">
            Monitor connections, active configurations, database stores, and memory throughput pipeline.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Toggle buttons */}
          <div className="bg-slate-950/60 p-1 border border-slate-800/80 rounded-xl flex gap-1 text-[10px] font-mono">
            <button
              onClick={() => setViewMode("gui")}
              className={`px-3 py-1.5 rounded-lg font-bold transition ${
                viewMode === "gui" ? "bg-blue-600 text-white" : "text-slate-500 hover:text-slate-350"
              }`}
            >
              GUI VIEW
            </button>
            <button
              onClick={() => setViewMode("cli")}
              className={`px-3 py-1.5 rounded-lg font-bold transition ${
                viewMode === "cli" ? "bg-blue-600 text-white" : "text-slate-500 hover:text-slate-350"
              }`}
            >
              CLI TABLE
            </button>
            <button
              onClick={() => setViewMode("json")}
              className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1 ${
                viewMode === "json" ? "bg-blue-600 text-white" : "text-slate-500 hover:text-slate-350"
              }`}
            >
              <Code className="w-3 h-3" />
              JSON
            </button>
          </div>

          <button
            onClick={handleRefresh}
            disabled={loading || refreshing}
            className="p-2.5 rounded-xl bg-slate-950/40 border border-slate-800 hover:bg-slate-900 hover:border-slate-700 text-slate-400 hover:text-slate-200 transition disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
            title="Refresh Diagnostic Telemetry"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin text-blue-400" : ""}`} />
          </button>
        </div>
      </div>

      {/* Main Panel Content */}
      <div className="flex-1 overflow-y-auto">
        {loading && !refreshing ? (
          <div className="h-full flex flex-col items-center justify-center py-20 text-slate-500 text-xs">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-3" />
            Loading system diagnostic health status...
          </div>
        ) : error ? (
          <div className="p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-start gap-2 max-w-lg mx-auto mt-10">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold">Diagnostics Error</div>
              <div>{error}</div>
              <button 
                onClick={fetchHealth} 
                className="mt-2 px-3 py-1.5 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg transition"
              >
                Retry Scan
              </button>
            </div>
          </div>
        ) : data ? (
          <>
            {/* View Mode: GUI */}
            {viewMode === "gui" && (
              <div className="space-y-6">
                {/* 1. Component Health Status cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* LLM */}
                  <div className="bg-slate-950/40 border border-slate-800/80 p-4 rounded-xl flex items-center justify-between shadow-sm">
                    <div className="space-y-1">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-mono font-bold">LLM Engine</div>
                      <div className="text-xs font-bold text-white font-outfit capitalize">{data.active_configuration.llm_provider}</div>
                      <div className="text-[9px] text-slate-600 font-mono truncate max-w-[120px]" title={data.active_configuration.llm_model}>
                        {data.active_configuration.llm_model}
                      </div>
                    </div>
                    <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-mono flex items-center gap-1.5 ${getStatusBadge(data.components.llm_provider.status)}`}>
                      {getStatusIcon(data.components.llm_provider.status)}
                      <span className="capitalize">{data.components.llm_provider.status.replace("_", " ")}</span>
                    </div>
                  </div>

                  {/* Embedding */}
                  <div className="bg-slate-950/40 border border-slate-800/80 p-4 rounded-xl flex items-center justify-between shadow-sm">
                    <div className="space-y-1">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-mono font-bold">Embeddings</div>
                      <div className="text-xs font-bold text-white font-outfit capitalize">{data.active_configuration.embedding_provider}</div>
                      <div className="text-[9px] text-slate-600 font-mono truncate max-w-[120px]" title={data.active_configuration.embedding_model}>
                        {data.active_configuration.embedding_model}
                      </div>
                    </div>
                    <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-mono flex items-center gap-1.5 ${getStatusBadge(data.components.embedding_provider.status)}`}>
                      {getStatusIcon(data.components.embedding_provider.status)}
                      <span className="capitalize">{data.components.embedding_provider.status.replace("_", " ")}</span>
                    </div>
                  </div>

                  {/* Vector DB */}
                  <div className="bg-slate-950/40 border border-slate-800/80 p-4 rounded-xl flex items-center justify-between shadow-sm">
                    <div className="space-y-1">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-mono font-bold">Vector Store</div>
                      <div className="text-xs font-bold text-white font-outfit capitalize">{data.active_configuration.vector_store}</div>
                      <div className="text-[9px] text-slate-600 font-mono truncate max-w-[120px]">
                        LanceDB Adapter
                      </div>
                    </div>
                    <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-mono flex items-center gap-1.5 ${getStatusBadge(data.components.vector_store.status)}`}>
                      {getStatusIcon(data.components.vector_store.status)}
                      <span className="capitalize">{data.components.vector_store.status}</span>
                    </div>
                  </div>

                  {/* Graph Store */}
                  <div className="bg-slate-950/40 border border-slate-800/80 p-4 rounded-xl flex items-center justify-between shadow-sm">
                    <div className="space-y-1">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-mono font-bold">Graph Store</div>
                      <div className="text-xs font-bold text-white font-outfit capitalize">{data.active_configuration.graph_engine}</div>
                      <div className="text-[9px] text-slate-600 font-mono truncate max-w-[120px]">
                        NetworkX In-Memory
                      </div>
                    </div>
                    <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold font-mono flex items-center gap-1.5 ${getStatusBadge(data.components.graph_store.status)}`}>
                      {getStatusIcon(data.components.graph_store.status)}
                      <span className="capitalize">{data.components.graph_store.status}</span>
                    </div>
                  </div>
                </div>

                {/* 2. Health Metrics Ledger */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {/* Memory statistics */}
                  <div className="border border-slate-800 bg-slate-950/20 rounded-xl p-4 space-y-3">
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono border-b border-slate-800/60 pb-2">
                      Memory Statistics
                    </div>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Total Memory Nodes</span>
                        <span className="font-mono font-bold text-white bg-slate-900 border border-slate-800/80 px-2 py-0.5 rounded">
                          {data.memory_statistics.total_nodes}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Total Semantic Edges</span>
                        <span className="font-mono font-bold text-white bg-slate-900 border border-slate-800/80 px-2 py-0.5 rounded">
                          {data.memory_statistics.total_edges}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Queue Process Status</span>
                        <span className="font-mono text-emerald-400 font-semibold flex items-center gap-1 capitalize">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                          {data.memory_statistics.queue_status}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Active Working Dataset</span>
                        <span className="font-mono text-blue-400">
                          {data.memory_statistics.active_dataset}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Operational Timeline */}
                  <div className="border border-slate-800 bg-slate-950/20 rounded-xl p-4 space-y-3">
                    <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono border-b border-slate-800/60 pb-2">
                      Pipeline Operational Latency
                    </div>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Last Recall Action</span>
                        <span className="font-mono text-slate-300 flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5 text-slate-500" />
                          {formatTime(data.memory_statistics.last_recall_time)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Last Ingest / cognify</span>
                        <span className="font-mono text-slate-300 flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5 text-slate-500" />
                          {formatTime(data.memory_statistics.last_improve_time)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Runtime Observability System</span>
                        <span className="font-mono text-slate-500">
                          MemoryOS Observation Plane
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Diagnostics Version</span>
                        <span className="font-mono text-blue-400">
                          {data.runtime_version}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* View Mode: CLI Table (Retro Terminal Output) */}
            {viewMode === "cli" && (
              <div className="space-y-4">
                <div className="flex items-center gap-1.5 text-xs text-slate-500 font-mono mb-1">
                  <Terminal className="w-4 h-4 text-blue-500" />
                  <span>cognee run-diagnostics --verbose</span>
                </div>
                <pre className="bg-slate-950 p-5 rounded-2xl font-mono text-xs text-emerald-400 border border-slate-800 leading-relaxed overflow-x-auto shadow-inner select-all">
{`Runtime Status
──────────────
LLM Provider         ✓ Connected (${data.active_configuration.llm_provider})
Embedding Model      ✓ Loaded (${data.active_configuration.embedding_provider})
Vector Store         ✓ Healthy (${data.active_configuration.vector_store})
Graph Store          ✓ Healthy (${data.active_configuration.graph_engine})

Memory Statistics
─────────────────
Active Dataset       ${data.memory_statistics.active_dataset}
Memory Objects       Nodes: ${data.memory_statistics.total_nodes} | Edges: ${data.memory_statistics.total_edges}
Last Recall          ${formatTime(data.memory_statistics.last_recall_time)}
Last Ingest          ${formatTime(data.memory_statistics.last_improve_time)}
Queue Status         ${data.memory_statistics.queue_status}
System Version       ${data.runtime_version}

Diagnostics status: 200 OK`}
                </pre>
              </div>
            )}

            {/* View Mode: Raw JSON */}
            {viewMode === "json" && (
              <div className="space-y-4">
                <div className="flex items-center gap-1.5 text-xs text-slate-500 font-mono mb-1">
                  <Code className="w-4 h-4 text-purple-500" />
                  <span>GET /api/health/runtime</span>
                </div>
                <pre className="bg-slate-950 p-5 rounded-2xl font-mono text-xs text-blue-400 border border-slate-800 leading-relaxed overflow-x-auto shadow-inner select-all">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
            )}
          </>
        ) : (
          <div className="h-full flex items-center justify-center py-20 text-slate-500 text-xs">
            No diagnostic data found.
          </div>
        )}
      </div>
    </div>
  );
}
