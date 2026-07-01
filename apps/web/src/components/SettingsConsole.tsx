"use client";

import React, { useState, useEffect } from "react";
import { Settings, ShieldCheck, AlertTriangle, Key, Loader2, Save } from "lucide-react";

export default function SettingsConsole() {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form Fields
  const [llmProvider, setLlmProvider] = useState("openai");
  const [llmModel, setLlmModel] = useState("gpt-4o");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [embeddingProvider, setEmbeddingProvider] = useState("openai");
  const [embeddingModel, setEmbeddingModel] = useState("text-embedding-3-large");
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [tenantKey, setTenantKey] = useState("");

  // DB Info (Readonly for local safety)
  const [dbInfo, setDbInfo] = useState({
    relational: "sqlite",
    graph: "networkx",
    vector: "lancedb"
  });

  const fetchSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/settings");
      const data = await res.json();
      if (!res.ok) throw new Error("Failed to fetch settings.");

      setLlmProvider(data.llm_provider || "openai");
      setLlmModel(data.llm_model || "gpt-4o");
      setEmbeddingProvider(data.embedding_provider || "openai");
      setEmbeddingModel(data.embedding_model || "text-embedding-3-large");
      
      // If keys are masked, don't auto-fill them so we don't overwrite with dots
      if (data.llm_api_key && !data.llm_api_key.startsWith("...")) {
        setLlmApiKey(data.llm_api_key);
      }
      if (data.gemini_api_key && !data.gemini_api_key.startsWith("...")) {
        setGeminiApiKey(data.gemini_api_key);
      }

      setDbInfo({
        relational: data.relational_database || "sqlite",
        graph: data.graph_database || "networkx",
        vector: data.vector_database || "lancedb"
      });

    } catch (err: any) {
      setError(err.message || "Failed to load backend configurations.");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setSuccess(null);

    const body: any = {
      llm_provider: llmProvider,
      llm_model: llmModel,
      embedding_provider: embeddingProvider,
      embedding_model: embeddingModel
    };

    if (llmApiKey && !llmApiKey.startsWith("...")) {
      body.llm_api_key = llmApiKey;
    }
    if (geminiApiKey && !geminiApiKey.startsWith("...")) {
      body.gemini_api_key = geminiApiKey;
    }

    // Auto-adjust embedding settings if switching provider
    if (llmProvider === "gemini") {
      body.llm_model = llmModel;
      body.embedding_model = "gemini-embedding-001";
    } else if (llmProvider === "openai") {
      body.embedding_provider = "openai";
      body.embedding_model = "text-embedding-3-large";
    }

    try {
      localStorage.setItem("memoryos_tenant_key", tenantKey);
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to update configurations.");

      setSuccess("Configuration settings saved successfully.");
      setTimeout(() => setSuccess(null), 3000);
      fetchSettings();
    } catch (err: any) {
      setError(err.message || "Could not save settings.");
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    fetchSettings();
    if (typeof window !== "undefined") {
      setTenantKey(localStorage.getItem("memoryos_tenant_key") || "");
    }
  }, []);

  const getMissingKeyWarning = () => {
    if (llmProvider === "openai" && !llmApiKey) {
      return "OpenAI API Key is missing. MemoryOS cannot run reasoning passes.";
    }
    if (llmProvider === "gemini" && !geminiApiKey) {
      return "Gemini API Key is missing. MemoryOS cannot run reasoning passes.";
    }
    return null;
  };

  const warning = getMissingKeyWarning();

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      <h2 className="text-lg font-bold text-white font-outfit mb-1">System Settings</h2>
      <p className="text-slate-500 text-xs mb-5">
        Configure API credentials, choose AI models, and inspect relational/vector/graph databases.
      </p>

      {loading ? (
        <div className="flex-1 flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
        </div>
      ) : (
        <form onSubmit={handleSave} className="flex-1 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            
            {/* AI Provider Config */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">
                  Reasoning LLM Provider
                </label>
                <select
                  value={llmProvider}
                  onChange={(e) => {
                    setLlmProvider(e.target.value);
                    setLlmModel(e.target.value === "gemini" ? "gemini-flash-latest" : "gpt-4o");
                  }}
                  className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full font-semibold text-slate-355"
                >
                  <option value="openai">OpenAI (Default)</option>
                  <option value="gemini">Google Gemini</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">
                  LLM Model Target
                </label>
                <input
                  type="text"
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                  className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full font-mono text-slate-300"
                />
              </div>
            </div>

            {/* API Keys */}
            <div className="space-y-3 pt-2">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono flex items-center gap-1">
                <Key className="w-3.5 h-3.5 text-blue-400" />
                API Credentials
              </div>

              <div className="grid grid-cols-1 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 font-medium">OpenAI API Key</label>
                  <input
                    type="password"
                    placeholder="sk-..."
                    value={llmApiKey}
                    onChange={(e) => setLlmApiKey(e.target.value)}
                    className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full font-mono text-slate-400"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 font-medium">Gemini API Key</label>
                  <input
                    type="password"
                    placeholder="AIzaSy..."
                    value={geminiApiKey}
                    onChange={(e) => setGeminiApiKey(e.target.value)}
                    className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full font-mono text-slate-400"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 font-medium">Tenant API Key (X-Tenant-Auth)</label>
                  <input
                    type="password"
                    placeholder="Enter Tenant Authorization Key..."
                    value={tenantKey}
                    onChange={(e) => setTenantKey(e.target.value)}
                    className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full font-mono text-slate-400"
                  />
                </div>
              </div>
            </div>

            {/* Databases Status */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono">
                Cognee Storage Adapters (Local Default)
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-[10px] font-mono">
                <div className="p-2.5 bg-slate-950/40 border border-slate-800 rounded-lg">
                  <div className="text-[8px] text-slate-600 uppercase font-bold">Relational</div>
                  <div className="text-slate-350 mt-0.5 truncate" title={dbInfo.relational}>
                    {dbInfo.relational.includes("sqlite") ? "SQLite DB" : "PostgreSQL"}
                  </div>
                </div>
                <div className="p-2.5 bg-slate-950/40 border border-slate-800 rounded-lg">
                  <div className="text-[8px] text-slate-600 uppercase font-bold">Graph</div>
                  <div className="text-slate-350 mt-0.5 capitalize">{dbInfo.graph}</div>
                </div>
                <div className="p-2.5 bg-slate-950/40 border border-slate-800 rounded-lg">
                  <div className="text-[8px] text-slate-600 uppercase font-bold">Vector</div>
                  <div className="text-slate-350 mt-0.5 capitalize">{dbInfo.vector}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Warning */}
          {warning && (
            <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 text-xs rounded-xl flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{warning}</span>
            </div>
          )}

          {success && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-xl flex items-start gap-2">
              <ShieldCheck className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{success}</span>
            </div>
          )}

          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={saving}
            className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-blue-800/50 text-white text-xs font-bold py-3 rounded-xl transition flex items-center justify-center gap-2 shadow shadow-blue-500/20"
          >
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            Save & Sync Cognee
          </button>
        </form>
      )}
    </div>
  );
}
