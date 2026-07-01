"use client";

import React, { useState } from "react";
import { UploadCloud, Link, FileText, CheckCircle, AlertTriangle, Loader2 } from "lucide-react";

interface IngestionConsoleProps {
  onIngestionSuccess: () => void;
  datasetName: string;
}

export default function IngestionConsole({ onIngestionSuccess, datasetName }: IngestionConsoleProps) {
  const [activeTab, setActiveTab] = useState<"text" | "file" | "url">("text");
  
  // Input fields
  const [text, setText] = useState("");
  const [textTitle, setTextTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [sessionId, setSessionId] = useState("");

  // States
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const resetMessages = () => {
    setError(null);
    setSuccess(null);
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    setLoading(true);
    resetMessages();
    
    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/ingest/text", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Tenant-Auth": tenantKey
        },
        body: JSON.stringify({
          text,
          dataset_name: datasetName,
          title: textTitle || "Raw Text Block",
          session_id: sessionId || undefined
        })
      });
      
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to ingest text memory.");
      
      setSuccess("Successfully ingested text block into AI memory.");
      setText("");
      setTextTitle("");
      onIngestionSuccess();
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    resetMessages();

    const formData = new FormData();
    formData.append("file", file);
    formData.append("dataset_name", datasetName);
    if (sessionId) {
      formData.append("session_id", sessionId);
    }

    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/ingest/file", {
        method: "POST",
        headers: {
          "X-Tenant-Auth": tenantKey
        },
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to ingest file memory.");

      setSuccess(`Successfully ingested "${file.name}" into AI memory.`);
      setFile(null);
      onIngestionSuccess();
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    resetMessages();

    try {
      const tenantKey = typeof window !== "undefined" ? (localStorage.getItem("memoryos_tenant_key") || "") : "";
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/ingest/url", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Tenant-Auth": tenantKey
        },
        body: JSON.stringify({
          url,
          dataset_name: datasetName,
          session_id: sessionId || undefined
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to scrape URL memory.");

      setSuccess(`Web scraped URL successfully and saved to memory.`);
      setUrl("");
      onIngestionSuccess();
    } catch (err: any) {
      setError(err.message || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      <h2 className="text-lg font-bold text-white font-outfit mb-1">Universal Memory Ingestion</h2>
      <p className="text-slate-500 text-xs mb-4">
        Ingest logs, articles, papers, or documents to grow the AI agents' long-term memory graph.
      </p>

      {/* Tabs */}
      <div className="flex bg-slate-950/60 p-1 rounded-xl border border-slate-800 mb-5 text-xs font-semibold">
        <button
          onClick={() => { setActiveTab("text"); resetMessages(); }}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition ${
            activeTab === "text"
              ? "bg-slate-800 text-white shadow shadow-black/50"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <FileText className="w-3.5 h-3.5" />
          Raw Text
        </button>
        <button
          onClick={() => { setActiveTab("file"); resetMessages(); }}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition ${
            activeTab === "file"
              ? "bg-slate-800 text-white shadow shadow-black/50"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <UploadCloud className="w-3.5 h-3.5" />
          Document Upload
        </button>
        <button
          onClick={() => { setActiveTab("url"); resetMessages(); }}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg transition ${
            activeTab === "url"
              ? "bg-slate-800 text-white shadow shadow-black/50"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Link className="w-3.5 h-3.5" />
          Scrape URL
        </button>
      </div>

      {/* Session ID Configuration (Universal) */}
      <div className="mb-4 space-y-1">
        <label className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono block">
          Session ID (Optional)
        </label>
        <input
          type="text"
          placeholder="e.g. agent_session_42"
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-2.5 w-full font-mono text-slate-400"
        />
      </div>

      {/* Forms Area */}
      <div className="flex-1">
        {/* Tab 1: Text */}
        {activeTab === "text" && (
          <form onSubmit={handleTextSubmit} className="space-y-3 h-full flex flex-col">
            <input
              type="text"
              placeholder="Memory Title (e.g. Meeting notes)"
              value={textTitle}
              onChange={(e) => setTextTitle(e.target.value)}
              className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full"
            />
            <textarea
              placeholder="Type or paste memory content here..."
              required
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full flex-1 min-h-[160px] resize-none leading-relaxed font-sans"
            />
            <div className="flex items-center justify-between text-[9px] text-slate-500 font-mono px-1">
              <span>Characters: {text.length}</span>
              <span>Est. Tokens: {Math.ceil(text.length / 4)}</span>
            </div>
            {Math.ceil(text.length / 4) > 1500 && (
              <div className="p-2.5 bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 text-[10px] rounded-xl flex items-start gap-1.5 leading-relaxed">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span>Large payload. Ingestion might run a retry sleep sequence to manage Groq/Gemini free-tier token limits.</span>
              </div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-blue-800/50 disabled:cursor-not-allowed text-white text-xs font-bold py-3 rounded-xl transition flex items-center justify-center gap-2 shadow shadow-blue-500/25"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? "Cognifying Synapse..." : "Remember Memory"}
            </button>
          </form>
        )}

        {/* Tab 2: File */}
        {activeTab === "file" && (
          <form onSubmit={handleFileSubmit} className="space-y-4 h-full flex flex-col justify-between">
            <div className="border border-dashed border-slate-800 hover:border-slate-700 rounded-2xl flex-1 flex flex-col items-center justify-center p-6 text-center cursor-pointer bg-slate-950/20 relative group transition">
              <input
                type="file"
                required
                accept=".pdf,.docx,.txt,.md,.json,.csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <UploadCloud className="w-10 h-10 text-slate-600 group-hover:text-slate-400 transition mb-3" />
              <h4 className="text-slate-300 font-semibold text-xs mb-1 font-outfit">
                {file ? file.name : "Select or drag document"}
              </h4>
              <p className="text-slate-600 text-[10px]">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : "Supports PDF, DOCX, TXT, MD up to 10MB"}
              </p>
            </div>
            <button
              type="submit"
              disabled={loading || !file}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-blue-800/50 disabled:cursor-not-allowed text-white text-xs font-bold py-3 rounded-xl transition flex items-center justify-center gap-2 shadow shadow-blue-500/25"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? "Extracting & Cognifying..." : "Upload & Parse"}
            </button>
          </form>
        )}

        {/* Tab 3: URL */}
        {activeTab === "url" && (
          <form onSubmit={handleUrlSubmit} className="space-y-4 h-full flex flex-col justify-between">
            <div className="space-y-2">
              <label className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono block">
                Webpage URL
              </label>
              <input
                type="url"
                required
                placeholder="https://example.com/api-documentation"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="bg-slate-950/50 border border-slate-800 focus:border-slate-700 outline-none text-xs rounded-xl p-3 w-full"
              />
              <p className="text-slate-600 text-[10px] leading-relaxed">
                MemoryOS will scrape the HTML content, clean layout scripts, strip footer headers, extract the main content, and index the data into Cognee.
              </p>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-blue-800/50 disabled:cursor-not-allowed text-white text-xs font-bold py-3 rounded-xl transition flex items-center justify-center gap-2 shadow shadow-blue-500/25"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? "Scraping & Indexing..." : "Scrape Webpage"}
            </button>
          </form>
        )}
      </div>

      {/* Notifications */}
      {success && (
        <div className="mt-4 p-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-xl flex items-start gap-2 shadow-sm animate-fadeIn">
          <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{success}</span>
        </div>
      )}
      {error && (
        <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-start gap-2 shadow-sm animate-fadeIn">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
