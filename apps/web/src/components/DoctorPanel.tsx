"use client";

import React, { useState, useEffect } from "react";
import { ShieldAlert, Check, RefreshCw, AlertTriangle, Info, ShieldCheck, Settings } from "lucide-react";

interface DiagnosticWarning {
  code: "ISOLATED_NODES" | "DUPLICATE_NODES" | "CONFLICTING_FACTS";
  severity: "LOW" | "MEDIUM" | "HIGH";
  message: string;
  details: any[];
}

interface ScanSummary {
  isolated_count: number;
  duplicate_pairs_count: number;
  conflict_count: number;
}

interface DoctorPanelProps {
  onMemoryFixed?: (duplicatePairs: any[]) => void;
}

export default function DoctorPanel({ onMemoryFixed }: DoctorPanelProps = {}) {
  const [healthIndex, setHealthIndex] = useState<number>(100);
  const [warnings, setWarnings] = useState<DiagnosticWarning[]>([]);
  const [summary, setSummary] = useState<ScanSummary>({
    isolated_count: 0,
    duplicate_pairs_count: 0,
    conflict_count: 0
  });

  const [scanning, setScanning] = useState(false);
  const [fixing, setFixing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastScanned, setLastScanned] = useState<string | null>(null);

  // Deterministic demo data seed
  useEffect(() => {
    const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
    fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + '/api/demo/seed-conflict', { 
      method: 'POST',
      headers: { "X-Tenant-Auth": tenantKey }
    })
      .catch(err => console.error('Failed to seed demo data:', err));
  }, []);

  const runScan = async (silent: boolean = false) => {
    if (!silent) setScanning(true);
    setError(null);
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/graph/diagnostics", {
        headers: { "X-Tenant-Auth": tenantKey }
      });
      const data = await res.json();
      if (!res.ok) {
        if (res.status === 503 && data.status === "busy") {
          throw new Error("Service busy, please retry");
        }
        throw new Error(data.detail || data.message || "Scan failed.");
      }
      
      let finalWarnings = data.warnings || [];
      let finalSummary = data.summary || { isolated_count: 0, duplicate_pairs_count: 0, conflict_count: 0 };
      let finalHealth = data.health_index;


      
      setHealthIndex(finalHealth);
      setWarnings(finalWarnings);
      setSummary(finalSummary);
      setLastScanned(new Date().toLocaleTimeString());
    } catch (err: any) {
      setError(err.message || "Could not complete diagnostics scan.");
    } finally {
      setScanning(false);
    }
  };

  const runFix = async () => {
    setFixing(true);
    setError(null);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/graph/diagnostics/fix", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Tenant-Auth": tenantKey
        },
        body: JSON.stringify({ fix_type: "all" }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      const data = await res.json();
      if (!res.ok) {
        if (res.status === 503 && data.status === "busy") {
          throw new Error("Service busy, please retry");
        }
        throw new Error(data.detail || data.message || "Optimization failed.");
      }
      
      const scan = data.scan_results || {};
      setHealthIndex(scan.health_index ?? 100);
      setWarnings(scan.warnings ?? []);
      setSummary(scan.summary ?? { isolated_count: 0, duplicate_pairs_count: 0, conflict_count: 0 });
      setLastScanned(new Date().toLocaleTimeString());
      
      // Extract duplicate pairs from BEFORE the fix to animate them
      // We read resolved_pairs explicitly provided by the backend, as the fresh scan results will have 0 duplicates.
      let duplicatePairs = data.resolved_pairs || [];
      

      
      if (onMemoryFixed) {
        setTimeout(() => {
          onMemoryFixed(duplicatePairs);
        }, 1500); // 1.5s delay to see the updated counts
      }
      
    } catch (err: any) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        setError("Repair timed out after 5 seconds. Graph may still be processing in the background.");
      } else {
        setError(err.message || "Failed to execute auto-fix.");
      }
    } finally {
      setFixing(false);
    }
  };

  // Run initial scan
  useEffect(() => {
    runScan(true);
  }, []);

  // Determine health bar gradient
  const getHealthColor = (h: number) => {
    if (h >= 90) return "text-emerald-500 shadow-emerald-500/25";
    if (h >= 70) return "text-yellow-500 shadow-yellow-500/25";
    return "text-rose-500 shadow-rose-500/25";
  };

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      <div className="flex justify-between items-start mb-1">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-white font-outfit">AI Memory Doctor</h2>
            <span className="bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 text-[10px] px-1.5 py-0.5 rounded font-mono uppercase tracking-wider">Simulation Mode</span>
          </div>
          <p className="text-slate-500 text-xs">
            Audit long-term cognitive states. Detect and resolve contradictions, duplicates, and dead loops.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={(e) => {
              e.currentTarget.disabled = true;
              runScan();
            }}
            disabled={scanning || fixing}
            className="bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-xs px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5 font-medium"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${scanning ? "animate-spin" : ""}`} />
            Scan
          </button>
          <button
            onClick={(e) => {
              e.currentTarget.disabled = true;
              runFix();
            }}
            disabled={scanning || fixing || healthIndex === 100}
            className="bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-50 text-white text-xs px-3.5 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 shadow shadow-blue-500/20"
          >
            <Settings className={`w-3.5 h-3.5 ${fixing ? "animate-spin" : ""}`} />
            Optimize Graph
          </button>
        </div>
      </div>

      {/* Main Diagnosis Split */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-6 overflow-hidden pt-4 min-h-[340px]">
        
        {/* Left: Health Meter */}
        <div className="border border-slate-800/80 rounded-xl p-5 bg-slate-950/20 flex flex-col items-center justify-center text-center">
          <div className="relative flex items-center justify-center mb-4">
            {/* Health circle glow */}
            <div className={`w-32 h-32 rounded-full border-4 border-slate-800 flex flex-col items-center justify-center shadow-lg relative`}>
              <span className={`text-4xl font-extrabold font-mono tracking-tighter ${getHealthColor(healthIndex)}`}>
                {healthIndex}
              </span>
              <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold mt-0.5">Health Index</span>
            </div>
          </div>

          <div className="w-full text-left space-y-2.5 font-mono text-[10px] text-slate-400 bg-slate-950/40 p-3 rounded-lg border border-slate-900">
            <div className="flex justify-between items-center">
              <span>Conflicts:</span>
              <span className={summary.conflict_count > 0 ? "text-rose-400 font-bold" : "text-slate-500"}>
                {summary.conflict_count} issues
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Duplicate Names:</span>
              <span className={summary.duplicate_pairs_count > 0 ? "text-yellow-400 font-bold" : "text-slate-500"}>
                {summary.duplicate_pairs_count} pairs
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>Isolated Vertices:</span>
              <span className={summary.isolated_count > 0 ? "text-slate-300 font-bold" : "text-slate-500"}>
                {summary.isolated_count} nodes
              </span>
            </div>
            <div className="border-t border-slate-900 pt-2 flex justify-between items-center text-[9px] text-slate-500">
              <span>Last Scan:</span>
              <span>{lastScanned || "Never"}</span>
            </div>
          </div>
        </div>

        {/* Right: Diagnosis Details */}
        <div className="md:col-span-2 border border-slate-800/80 rounded-xl p-4 bg-slate-950/30 overflow-y-auto h-full flex flex-col justify-between">
          <div className="space-y-3 flex-1 overflow-y-auto">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">
              Diagnostic Audit Report
            </div>

            {warnings.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                <ShieldCheck className="w-10 h-10 text-emerald-500 mb-2 animate-bounce" />
                <span className="text-xs font-bold text-slate-300">Memory Graph Healthy</span>
                <span className="text-[10px] text-slate-600 mt-0.5">No contradictions or isolated nodes discovered.</span>
              </div>
            ) : (
              warnings.map((w, idx) => (
                <div
                  key={idx}
                  className={`p-3 border rounded-xl space-y-2 ${
                    w.severity === "HIGH"
                      ? "bg-rose-500/5 border-rose-500/20 text-rose-200"
                      : w.severity === "MEDIUM"
                        ? "bg-yellow-500/5 border-yellow-500/20 text-yellow-200"
                        : "bg-slate-500/5 border-slate-800 text-slate-300"
                  }`}
                >
                  <div className="flex items-center gap-1.5 text-xs font-bold">
                    {w.severity === "HIGH" ? (
                      <ShieldAlert className="w-4 h-4 text-rose-500" />
                    ) : w.severity === "MEDIUM" ? (
                      <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    ) : (
                      <Info className="w-4 h-4 text-slate-400" />
                    )}
                    <span>{w.message}</span>
                  </div>

                  {/* Warning details */}
                  <div className="font-mono text-[9px] pl-5 space-y-1 text-slate-400">
                    {w.code === "CONFLICTING_FACTS" &&
                      w.details.map((d: any, dIdx) => (
                        <div key={dIdx} className="bg-black/35 p-2 rounded border border-rose-500/10">
                          Conflict: <span className="text-rose-400">{d.source.label}</span> has multiple values for relation{" "}
                          <span className="font-bold text-white">[{d.relationship}]</span> pointing to:{" "}
                          <span className="text-slate-300">{d.targets.map((t: any) => t.label).join(", ")}</span>
                        </div>
                      ))}
                    {w.code === "DUPLICATE_NODES" &&
                      w.details.map((d: any, dIdx) => (
                        <div key={dIdx} className="bg-black/35 p-2 rounded border border-yellow-500/10 flex justify-between">
                          <span>
                            Similarity Match: <span className="text-yellow-400">"{d.node1.label}"</span> vs{" "}
                            <span className="text-yellow-400">"{d.node2.label}"</span>
                          </span>
                          <span className="text-yellow-500 font-bold">{Math.round(d.similarity * 100)}%</span>
                        </div>
                      ))}
                    {w.code === "ISOLATED_NODES" && (
                      <div className="flex flex-wrap gap-1.5">
                        {w.details.map((d: any, dIdx) => (
                          <span key={dIdx} className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 text-[8px]">
                            {d.label}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Action Cues */}
          {healthIndex < 100 && !fixing && (
            <div className="mt-4 p-3 bg-blue-500/5 border border-blue-500/20 text-blue-400 text-xs rounded-xl flex items-center gap-2">
              <Check className="w-4 h-4" />
              <span>Optimizing the graph will invoke Cognee's <code>improve()</code> algorithms to merge synonyms and prune isolated loops.</span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-center gap-2">
          <ShieldAlert className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
