"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Compass, 
  UploadCloud, 
  Brain, 
  Activity, 
  Users, 
  Settings as SettingsIcon, 
  Trash2, 
  Database,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Menu,
  X,
  HeartPulse
} from "lucide-react";

import GalaxyVisualizer from "../components/GalaxyVisualizer";
import MemoryDNA from "../components/MemoryDNA";
import IngestionConsole from "../components/IngestionConsole";
import RecallConsole from "../components/RecallConsole";
import DoctorPanel from "../components/DoctorPanel";
import AgentWorkspace from "../components/AgentWorkspace";
import TimelineConsole from "../components/TimelineConsole";
import SettingsConsole from "../components/SettingsConsole";
import HealthDashboard from "../components/HealthDashboard";

interface Node {
  id: string;
  label: string;
  type: string;
  properties: any;
  heat: number;
  dna: any;
}

interface Edge {
  source: string;
  target: string;
  label: string;
  properties: any;
}

interface GraphMetrics {
  total_nodes: number;
  total_edges: number;
  density: number;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"galaxy" | "ingest" | "recall" | "doctor" | "agents" | "settings" | "health">("galaxy");
  
  // Graph States
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [metrics, setMetrics] = useState<GraphMetrics>({ total_nodes: 0, total_edges: 0, density: 0 });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  
  // Custom HUD states
  const [heatmapEnabled, setHeatmapEnabled] = useState(false);
  const [activePathNodes, setActivePathNodes] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState("main_dataset");
  const [timeTravelTimestamp, setTimeTravelTimestamp] = useState<string | null>(null);
  
  // Repair animation states
  const [isRepairing, setIsRepairing] = useState(false);
  const [repairTargets, setRepairTargets] = useState<any[]>([]);
  const repairTimeoutRef = useRef<any>(null);
  
  // Triggers for data refreshes
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [apiOnline, setApiOnline] = useState(false);
  const [healthIndex, setHealthIndex] = useState(100);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleMemoryFixed = (duplicatePairs: any[]) => {
    if (repairTimeoutRef.current) clearTimeout(repairTimeoutRef.current);
    setRepairTargets(duplicatePairs);
    setActiveTab("galaxy");
    setIsRepairing(true);
    triggerRefresh();
    
    // Stop repair animation after 4 seconds
    repairTimeoutRef.current = setTimeout(() => {
      setIsRepairing(false);
      setRepairTargets([]);
      repairTimeoutRef.current = null;
    }, 4000);
  };

  // Fetch graph data from backend
  const fetchGraphData = async () => {
    setIsLoadingGraph(true);
    try {
      let url = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/graph/data";
      if (timeTravelTimestamp) {
        url += `?timestamp=${encodeURIComponent(timeTravelTimestamp)}`;
      }
      
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch(url, {
        headers: { "X-Tenant-Auth": tenantKey }
      });
      if (!res.ok) throw new Error("API failed");
      
      const data = await res.json();
      setNodes(data.nodes || []);
      setEdges(data.edges || []);
      setMetrics(data.metrics || { total_nodes: 0, total_edges: 0, density: 0 });
      setApiOnline(true);
      
      // Update selected node properties if it still exists
      if (selectedNode) {
        const updated = (data.nodes || []).find((n: Node) => n.id === selectedNode.id);
        setSelectedNode(updated || null);
      }
    } catch (e) {
      setApiOnline(false);
    } finally {
      setIsLoadingGraph(false);
    }
  };

  // Fetch health stats
  const fetchHealthIndex = async () => {
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/graph/diagnostics", {
        headers: { "X-Tenant-Auth": tenantKey }
      });
      if (res.ok) {
        const data = await res.json();
        setHealthIndex(data.health_index);
      }
    } catch (e) {}
  };

  useEffect(() => {
    fetchGraphData();
    fetchHealthIndex();
  }, [refreshTrigger, timeTravelTimestamp]);

  // Handle dataset swaps (clear state when switching datasets)
  useEffect(() => {
    setSelectedNode(null);
    setActivePathNodes([]);
    setTimeTravelTimestamp(null);
    setRefreshTrigger(prev => prev + 1);
  }, [datasetName]);

  const triggerRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
    fetchHealthIndex();
  };

  const handleClearMemory = async () => {
    if (!confirm("Are you sure you want to delete all memories? This wipes the Cognee database.")) return;
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/memory/clear?everything=true", {
        method: "POST",
        headers: { "X-Tenant-Auth": tenantKey }
      });
      if (res.ok) {
        alert("All memories wiped successfully.");
        triggerRefresh();
        setSelectedNode(null);
      }
    } catch (e) {
      alert("Failed to clear memory.");
    }
  };

  return (
    <main className="w-full min-h-screen bg-slate-950 text-slate-100 flex font-sans overflow-hidden">
      
      {/* Mobile Sidebar Backdrop */}
      {isSidebarOpen && (
        <button
          onClick={() => setIsSidebarOpen(false)}
          className="fixed inset-0 z-45 bg-black/60 backdrop-blur-sm lg:hidden cursor-default w-full h-full border-none outline-none"
          aria-label="Close Navigation Sidebar"
        />
      )}

      {/* 1. LEFT SIDEBAR NAVIGATION */}
      <section className={`fixed lg:static inset-y-0 left-0 z-50 lg:z-auto w-64 border-r border-slate-900 bg-slate-950/80 backdrop-blur-md flex flex-col justify-between p-4 shrink-0 transition-transform duration-300 ${
        isSidebarOpen ? "translate-x-0" : "-translate-x-full"
      } lg:translate-x-0`}>
        <div className="space-y-6">
          
          {/* Logo Header */}
          <div className="px-2 py-3 border-b border-slate-900 flex items-center justify-between gap-2.5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 shadow-md shadow-blue-500/20 flex items-center justify-center font-bold font-outfit text-white">
                M
              </div>
              <div>
                <h1 className="text-sm font-black font-outfit tracking-wider text-white leading-none">MemoryOS</h1>
                <span className="text-[9px] font-mono font-bold tracking-widest text-blue-500 uppercase">AI Memory Layer</span>
              </div>
            </div>
            {/* Close Button for Mobile Drawer */}
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="lg:hidden p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 transition"
              aria-label="Close Navigation Sidebar"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            <button
              onClick={() => { setActiveTab("galaxy"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "galaxy"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <Compass className="w-4 h-4 text-blue-400" />
              Memory Galaxy
            </button>
            <button
              onClick={() => { setActiveTab("ingest"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "ingest"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <UploadCloud className="w-4 h-4 text-cyan-400" />
              Ingestion Console
            </button>
            <button
              onClick={() => { setActiveTab("recall"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "recall"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <Brain className="w-4 h-4 text-purple-400" />
              Search & Recall
            </button>
            <button
              onClick={() => { setActiveTab("doctor"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "doctor"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <Activity className="w-4 h-4 text-emerald-400" />
              AI Memory Doctor
            </button>
            <button
              onClick={() => { setActiveTab("agents"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "agents"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <Users className="w-4 h-4 text-amber-400" />
              Agent Collaboration
            </button>
            <button
              onClick={() => { setActiveTab("health"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "health"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <HeartPulse className="w-4 h-4 text-rose-500" />
              Runtime Health
            </button>
            <button
              onClick={() => { setActiveTab("settings"); setIsSidebarOpen(false); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition ${
                activeTab === "settings"
                  ? "bg-slate-900 text-white shadow shadow-black/35 border border-slate-800/80"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/30"
              }`}
            >
              <SettingsIcon className="w-4 h-4 text-slate-400" />
              System Settings
            </button>
          </nav>
        </div>

        {/* Footer Actions */}
        <div className="space-y-3 px-2">
          <div className="flex items-center justify-between text-[10px] font-mono">
            <span className="text-slate-600">Database API:</span>
            <span className={`font-bold flex items-center gap-1 ${apiOnline ? "text-emerald-500" : "text-rose-500 animate-pulse"}`}>
              <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
              {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>

          <button
            onClick={handleClearMemory}
            className="w-full bg-slate-950/80 hover:bg-rose-950/20 text-slate-500 hover:text-rose-400 border border-slate-900 hover:border-rose-900/50 py-2.5 rounded-xl text-xs font-semibold transition flex items-center justify-center gap-2"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Wipe Memory
          </button>
        </div>
      </section>

      {/* 2. MAIN HEADER & DYNAMIC VIEWS CONTAINER */}
      <section className="flex-1 flex flex-col min-w-0 bg-slate-950 bg-grid-pattern relative overflow-y-auto">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(59,130,246,0.03)_0%,transparent_70%)] pointer-events-none z-0" />
        
        {/* Top Statistics HUD */}
        <header className="border-b border-slate-900 p-4 bg-slate-950/60 backdrop-blur-md flex items-center justify-between gap-4 sticky top-0 z-20">
          
          <div className="flex items-center gap-3">
            {/* Mobile Hamburger Button */}
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden p-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 transition"
              aria-label="Open Navigation Sidebar"
            >
              <Menu className="w-4 h-4" />
            </button>
          </div>
          
          {/* Node/Edge count metrics */}
          <div className="flex flex-wrap gap-4 text-xs font-mono">
            <div className="flex items-center gap-2 bg-slate-900/30 px-3 py-1.5 border border-slate-900 rounded-xl">
              <Database className="w-3.5 h-3.5 text-blue-500" />
              <span className="text-slate-500">Nodes:</span>
              <span className="text-slate-200 font-bold">{metrics.total_nodes}</span>
            </div>
            <div className="flex items-center gap-2 bg-slate-900/30 px-3 py-1.5 border border-slate-900 rounded-xl">
              <Database className="w-3.5 h-3.5 text-cyan-500" />
              <span className="text-slate-500">Edges:</span>
              <span className="text-slate-200 font-bold">{metrics.total_edges}</span>
            </div>
            
            {/* Health badge */}
            <div className="flex items-center gap-2 bg-slate-900/30 px-3 py-1.5 border border-slate-900 rounded-xl">
              {healthIndex >= 90 ? (
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 text-yellow-500 animate-pulse" />
              )}
              <span className="text-slate-500">Memory Health:</span>
              <span className={`font-bold ${healthIndex >= 90 ? "text-emerald-400" : "text-yellow-500"}`}>
                {healthIndex}%
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Refresh button */}
            <button
              onClick={triggerRefresh}
              className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 transition"
              title="Refresh Graph Data"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>

            {/* Workspace Selector */}
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-slate-500 font-medium">Space:</span>
              <select
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-1.5 outline-none font-semibold text-slate-200 font-outfit"
              >
                <option value="main_dataset">Personal Space</option>
                <option value="agent_collab_space">Agent Team Space</option>
              </select>
            </div>
          </div>
        </header>

        {/* Tab Panels */}
        <section className="flex-1 p-6 overflow-y-auto relative z-10">
          {activeTab === "galaxy" && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full items-stretch">
              
              {/* Left 3/4: Canvas graph + Playback Time Machine */}
              <div className="lg:col-span-3 flex flex-col gap-6">
                
                {/* Galaxy HUD Controls */}
                <div className="flex justify-between items-center bg-slate-900/20 border border-slate-800/80 px-4 py-2.5 rounded-xl shadow-sm">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-bold text-white font-outfit">Memory Galaxy Map</span>
                  </div>
                  
                  {/* Heatmap Toggle */}
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-400">Temperature Heatmap:</span>
                    <button
                      onClick={() => setHeatmapEnabled(!heatmapEnabled)}
                      role="switch"
                      aria-checked={heatmapEnabled}
                      aria-label="Toggle Temperature Heatmap"
                      className={`relative w-10 h-5.5 rounded-full transition-colors duration-200 border ${
                        heatmapEnabled ? "bg-blue-600 border-blue-500" : "bg-slate-950 border-slate-800"
                      }`}
                    >
                      <span className={`absolute top-0.5 left-0.5 w-4.5 h-4.5 rounded-full bg-white transition-transform duration-200 ${
                        heatmapEnabled ? "translate-x-4.5" : "translate-x-0"
                      }`}></span>
                    </button>
                  </div>
                </div>

                {/* Main Graph Canvas */}
                <div className="flex-1 min-h-[460px] relative">
                  <GalaxyVisualizer
                    nodes={nodes}
                    edges={edges}
                    selectedNodeId={selectedNode?.id || null}
                    onSelectNode={setSelectedNode}
                    heatmapEnabled={heatmapEnabled}
                    activePathNodes={activePathNodes}
                    isRepairing={isRepairing}
                    repairTargets={repairTargets}
                  />
                  {nodes.length === 0 && apiOnline && !isLoadingGraph && (
                    <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-8 text-center rounded-2xl border border-slate-800 pointer-events-none">
                      <Brain className="w-12 h-12 text-slate-850 mb-3 animate-pulse" />
                      <h3 className="text-slate-400 font-bold mb-1 font-outfit text-sm">Memory Graph is Empty</h3>
                      <p className="text-slate-500 text-xs max-w-xs">
                        Ingest documents, paste text blocks, or run the Multi-Agent simulation to begin building memory.
                      </p>
                    </div>
                  )}
                  {isLoadingGraph && (
                    <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur border border-blue-500/30 text-blue-400 px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 pointer-events-none shadow-lg shadow-blue-900/20 z-10">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Syncing Graph...
                    </div>
                  )}
                  {!apiOnline && (
                    <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-8 text-center rounded-2xl border border-slate-850 pointer-events-none">
                      <AlertTriangle className="w-12 h-12 text-rose-500/80 mb-3 animate-pulse" />
                      <h3 className="text-rose-400 font-bold mb-1 font-outfit text-sm">FastAPI Offline</h3>
                      <p className="text-slate-500 text-xs max-w-xs leading-relaxed">
                        Start the Python FastAPI backend by running the launcher script: <code>./run_dev.ps1</code>
                      </p>
                    </div>
                  )}
                </div>

                {/* Time Machine Playback slider */}
                <div className="h-fit">
                  <TimelineConsole 
                    onTimeTravel={setTimeTravelTimestamp} 
                    refreshTrigger={refreshTrigger} 
                  />
                </div>
              </div>

              {/* Right 1/4: Selected DNA info */}
              <div className="lg:col-span-1 h-full min-h-[480px]">
                <MemoryDNA
                  node={selectedNode}
                  onClearSelection={() => setSelectedNode(null)}
                />
              </div>
            </div>
          )}

          {activeTab === "ingest" && (
            <div className="max-w-3xl mx-auto h-full">
              <IngestionConsole 
                onIngestionSuccess={triggerRefresh} 
                datasetName={datasetName} 
              />
            </div>
          )}

          {activeTab === "recall" && (
            <div className="max-w-5xl mx-auto h-full">
              <RecallConsole 
                onRecallCompleted={setActivePathNodes} 
                datasetName={datasetName} 
              />
            </div>
          )}

          {activeTab === "doctor" && (
            <div className="max-w-4xl mx-auto h-full">
              <DoctorPanel onMemoryFixed={handleMemoryFixed} />
            </div>
          )}

          {activeTab === "agents" && (
            <div className="max-w-5xl mx-auto h-full">
              <AgentWorkspace onAgentStepExecuted={setActivePathNodes} />
            </div>
          )}

          {activeTab === "health" && (
            <div className="max-w-4xl mx-auto h-full">
              <HealthDashboard />
            </div>
          )}

          {activeTab === "settings" && (
            <div className="max-w-2xl mx-auto h-full">
              <SettingsConsole />
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
