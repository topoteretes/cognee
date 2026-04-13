"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCogniInstance } from "@/modules/tenant/TenantProvider";
import { useFilter } from "@/ui/layout/FilterContext";
import getDatasets from "@/modules/datasets/getDatasets";
import searchDataset from "@/modules/datasets/searchDataset";
import addData from "@/modules/ingestion/addData";
import cognifyDataset from "@/modules/datasets/cognifyDataset";
import createDataset from "@/modules/datasets/createDataset";
import { notifications } from "@mantine/notifications";

interface PipelineRun { id: string; pipeline_name: string; status: string; dataset_id: string | null; dataset_name: string | null; owner_email: string | null; created_at: string | null; pipeline_run_id: string | null }

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function pipelineLabel(name: string): string {
  if (name.includes("cognify")) return "cognee.cognify";
  if (name.includes("add")) return "cognee.add";
  if (name.includes("search")) return "cognee.search";
  return name;
}

function ownerDisplayName(email: string | null): string {
  if (!email) return "System";
  if (email.endsWith("@cognee.agent")) {
    const local = email.split("@")[0];
    const parts = local.split("-");
    const shortId = parts.pop() || "";
    const type = parts.join(" ");
    return `${type} ${shortId}`;
  }
  if (email === "default_user@example.com") return "You";
  return email.split("@")[0];
}

function statusDot(status: string): string {
  if (status.includes("COMPLETED")) return "#22C55E";
  if (status.includes("ERRORED")) return "#EF4444";
  if (status.includes("STARTED") || status.includes("INITIATED")) return "#F59E0B";
  return "#A1A1AA";
}

export default function OverviewPage() {
  const { cogniInstance, isInitializing } = useCogniInstance();
  const { agents, datasets, selectedAgent, selectedDataset, loading: filterLoading } = useFilter();
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [traceCount, setTraceCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  async function handleDashboardUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!cogniInstance || !e.target.files?.length) return;
    const files = Array.from(e.target.files);
    e.target.value = "";
    setIsUploading(true);
    try {
      let ds = datasets[0];
      if (!ds) {
        ds = await createDataset({ name: "default_dataset" }, cogniInstance);
      }
      await addData({ id: ds.id, name: ds.name }, files, cogniInstance);
      notifications.show({ title: "Files uploaded — building knowledge graph...", message: `${files.length} file(s) added. Cognify running.`, color: "blue", autoClose: 5000 });
      await cognifyDataset({ id: ds.id, name: ds.name, data: [], status: "" }, cogniInstance);
      notifications.show({ title: "Knowledge graph built", message: `"${ds.name}" is now searchable.`, color: "green" });
    } catch (err) {
      console.error("Dashboard upload failed:", err);
      notifications.show({ title: "Upload failed", message: err instanceof Error ? err.message : String(err), color: "red" });
    } finally {
      setIsUploading(false);
    }
  }

  useEffect(() => {
    if (!cogniInstance || isInitializing) return;
    Promise.all([
      cogniInstance.fetch("/v1/activity/pipeline-runs").then((r) => r.ok ? r.json() : []).catch(() => []),
      cogniInstance.fetch("/v1/activity/spans").then((r) => r.ok ? r.json() : []).catch(() => []),
    ]).then(([runData, spanData]) => {
      setRuns(Array.isArray(runData) ? runData : []);
      setTraceCount(Array.isArray(spanData) ? spanData.length : 0);
      // Only redirect to onboarding if user has never dismissed it
      if (datasets.length === 0 && !filterLoading && !sessionStorage.getItem("cognee-onboarding-skipped")) {
        router.replace("/onboarding");
      }
    }).finally(() => setLoading(false));
  }, [cogniInstance, isInitializing, datasets, filterLoading, router]);

  if (loading || isInitializing || filterLoading) {
    return <div style={{ padding: 32, display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}><span style={{ fontSize: 14, color: "#71717A" }}>Loading...</span></div>;
  }

  // Deduplicate runs
  const latestRuns: PipelineRun[] = [];
  const seen = new Set<string>();
  for (const r of runs) {
    const key = r.pipeline_run_id || r.id;
    if (!seen.has(key)) { seen.add(key); latestRuns.push(r); }
  }

  // Filter runs by selected dataset if any
  const filteredRuns = selectedDataset
    ? latestRuns.filter((r) => r.dataset_id === selectedDataset.id)
    : latestRuns;

  const filteredDatasets = selectedDataset
    ? datasets.filter((d) => d.id === selectedDataset.id)
    : datasets;

  const agentCount = agents.filter((a) => a.is_agent).length;
  const apiCalls = filteredRuns.length + traceCount;
  const errorCount = filteredRuns.filter((r) => r.status.includes("ERRORED")).length;
  const errorRate = apiCalls > 0 ? ((errorCount / apiCalls) * 100).toFixed(1) : "0";

  const stats = [
    { label: "Connected Agents", value: String(agentCount), dot: agentCount > 0 ? "#22C55E" : undefined },
    { label: "Datasets", value: String(filteredDatasets.length) },
    { label: "API Calls (24h)", value: String(apiCalls) },
    { label: "Error Rate", value: `${errorRate}%`, color: Number(errorRate) > 5 ? "#EF4444" : "#22C55E" },
  ];

  return (
    <div style={{ padding: 32, display: "flex", flexDirection: "column", gap: 28, fontFamily: '"Inter", system-ui, sans-serif' }}>
      {/* Hidden file input for dashboard upload */}
      <input ref={uploadInputRef} type="file" multiple accept=".pdf,.csv,.txt,.md,.json,.docx" className="hidden" onChange={handleDashboardUpload} />

      {/* Context indicator */}
      {(selectedAgent || selectedDataset) && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#A1A1AA" }}>Showing data for:</span>
          {selectedAgent && <span style={{ background: "#F0EDFF", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{selectedAgent.agent_type}</span>}
          {selectedDataset && <span style={{ background: "#F0EDFF", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 500, color: "#6510F4" }}>{selectedDataset.name}</span>}
        </div>
      )}

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 16 }}>
        {stats.map((s) => (
          <div key={s.label} style={{ flex: 1, background: "#fff", border: "1px solid #EEEEEE", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 11, fontWeight: 500, letterSpacing: 0.5, color: "#999999", textTransform: "uppercase" }}>{s.label}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {s.dot && <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.dot, flexShrink: 0 }} />}
              <span style={{ fontSize: 20, fontWeight: 600, color: s.color || "#1A1A1A" }}>{s.value}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Get started */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: "#1A1A1A" }}>Get started</span>
        <div style={{ display: "flex", gap: 12 }}>
          <SdkCard />
          <Link href="/api-keys" className="cursor-pointer hover:bg-cognee-hover transition-colors" style={{ flex: 1, background: "#fff", border: "1px solid #EEEEEE", borderRadius: 10, padding: "20px 16px", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textDecoration: "none" }}>
            <KeyNavIcon />
            <span style={{ fontSize: 13, fontWeight: 500, color: "#333333" }}>API Key</span>
            <span style={{ fontSize: 11, color: "#999999" }}>Copy your key</span>
          </Link>
          <Link href="/connections" className="cursor-pointer hover:bg-cognee-hover transition-colors" style={{ flex: 1, background: "#fff", border: "1px solid #EEEEEE", borderRadius: 10, padding: "20px 16px", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textDecoration: "none" }}>
            <AgentNavIcon />
            <span style={{ fontSize: 13, fontWeight: 500, color: "#333333" }}>Agents</span>
            <span style={{ fontSize: 11, color: "#999999" }}>{agentCount} connected</span>
          </Link>
          <button onClick={() => uploadInputRef.current?.click()} className="cursor-pointer hover:bg-cognee-hover transition-colors" style={{ flex: 1, background: isUploading ? "#F5F3FF" : "#fff", border: isUploading ? "1px solid #D4D0F8" : "1px solid #EEEEEE", borderRadius: 10, padding: "20px 16px", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, textDecoration: "none" }}>
            {isUploading ? (
              <>
                <div style={{ width: 24, height: 24, border: "2px solid", borderColor: "#6510F4 transparent #6510F4 transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                <span style={{ fontSize: 13, fontWeight: 500, color: "#6510F4" }}>Processing...</span>
                <span style={{ fontSize: 11, color: "#7C3AED" }}>Building knowledge graph</span>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </>
            ) : (
              <>
                <UploadIcon />
                <span style={{ fontSize: 13, fontWeight: 500, color: "#333333" }}>Upload Data</span>
                <span style={{ fontSize: 11, color: "#999999" }}>Click to select files</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Search */}
      <DashboardSearch datasets={filteredDatasets} cogniInstance={cogniInstance} />

      {/* Recent activity */}
      {filteredRuns.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 15, fontWeight: 600, color: "#1A1A1A" }}>Recent activity</span>
            <Link href="/activity" style={{ fontSize: 12, color: "#6C47FF", textDecoration: "none" }}>View all</Link>
          </div>
          <div style={{ borderLeft: "2px solid #EEEEEE", paddingLeft: 20 }}>
            {filteredRuns.slice(0, 6).map((r) => {
              const dsName = r.dataset_name || datasets.find((d) => d.id === r.dataset_id)?.name || r.dataset_id?.slice(0, 8) || "unknown";
              const dot = statusDot(r.status);
              const agent = ownerDisplayName(r.owner_email);
              return (
                <div key={r.id} style={{ display: "flex", gap: 14, padding: "10px 0", position: "relative" }}>
                  <div style={{ position: "absolute", left: -25, top: 14, width: 10, height: 10, borderRadius: "50%", background: dot, border: "2px solid #FAFAF9" }} />
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color: "#1A1A1A" }}>{agent}</span>
                      <span style={{ fontSize: 12, fontWeight: 500, color: dot === "#EF4444" ? "#EF4444" : "#6C47FF" }}>{pipelineLabel(r.pipeline_name)}</span>
                      <span style={{ fontSize: 12, color: r.status.includes("ERRORED") ? "#EF4444" : "#A1A1AA" }}>
                        {r.status.includes("COMPLETED") ? "completed" : r.status.includes("STARTED") ? "started" : r.status.includes("ERRORED") ? "error" : r.status.toLowerCase()}
                      </span>
                    </div>
                    <span style={{ fontSize: 12, color: "#999999" }}>Dataset: {dsName}</span>
                  </div>
                  <span style={{ fontSize: 11, color: "#BBBBBB", flexShrink: 0 }}>{r.created_at ? timeAgo(r.created_at) : ""}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Inline dashboard search ──

function DashboardSearch({ datasets, cogniInstance }: { datasets: { id: string; name: string }[]; cogniInstance: ReturnType<typeof useCogniInstance>["cogniInstance"] }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ search_result: string[]; dataset_name: string }[]>([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  async function handleSearch(q: string) {
    if (!q.trim() || !cogniInstance) return;
    setSearching(true);
    setResults([]);
    setHasSearched(true);
    try {
      const data = await searchDataset(cogniInstance, {
        query: q,
        searchType: "GRAPH_COMPLETION",
        datasetIds: datasets.map((d) => d.id),
      });
      setResults(Array.isArray(data) ? data : []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: 200 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: "#1A1A1A" }}>Search your knowledge</span>
        <Link href="/search" style={{ fontSize: 12, color: "#6C47FF", textDecoration: "none" }}>Full search</Link>
      </div>
      <div style={{ background: "#fff", border: `1px solid ${query ? "#6510F4" : "#EEEEEE"}`, borderRadius: 10, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10, transition: "border-color 0.2s" }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" style={{ flexShrink: 0 }}>
          <circle cx="8" cy="8" r="5.5" stroke="#A1A1AA" strokeWidth="1.5" />
          <path d="M12.5 12.5L16 16" stroke="#A1A1AA" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") handleSearch(query); }}
          placeholder="Ask a question about your data..."
          style={{ flex: 1, border: "none", outline: "none", fontSize: 14, color: "#18181B", fontFamily: "inherit", background: "transparent" }}
        />
        {query && (
          <button onClick={() => handleSearch(query)} className="cursor-pointer" style={{ background: "#6510F4", border: "none", borderRadius: 6, padding: "6px 14px", fontSize: 13, fontWeight: 500, color: "#fff" }}>Search</button>
        )}
      </div>

      {searching && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 0" }}>
          <div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid #E4E4E7", borderTopColor: "#6510F4", animation: "spin 1s linear infinite" }} />
          <span style={{ fontSize: 13, color: "#71717A" }}>Searching...</span>
        </div>
      )}

      {results.length > 0 && !searching && (
        <div style={{ background: "#fff", border: "1px solid #E5E7EB", borderRadius: 10, overflow: "hidden" }}>
          {results.map((r, i) => (
            <div key={i} style={{ padding: "14px 16px", borderBottom: i < results.length - 1 ? "1px solid #F4F4F5" : "none" }}>
              <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.04em", color: "#6510F4", textTransform: "uppercase" }}>{r.dataset_name || "Result"}</span>
              {r.search_result.map((text, j) => (
                <p key={j} style={{ fontSize: 13, color: "#18181B", lineHeight: "20px", margin: "4px 0 0" }}>{text}</p>
              ))}
            </div>
          ))}
        </div>
      )}

      {hasSearched && !searching && results.length === 0 && (
        <span style={{ fontSize: 13, color: "#A1A1AA", padding: "8px 0" }}>No results found.</span>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Python SDK card with expandable connect-to-cloud instructions ──

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
      className="cursor-pointer hover:bg-white/10 rounded p-1 -m-1 active:scale-90 transition-all"
      style={{ background: "none", border: "none", flexShrink: 0 }}
      title="Copy"
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="#22C55E" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="8" height="8" rx="1.5" stroke="#71717A" strokeWidth="1.5" /><path d="M11 3H4.5A1.5 1.5 0 003 4.5V11" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" /></svg>
      )}
    </button>
  );
}

function SdkCard() {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className="cursor-pointer hover:bg-cognee-hover transition-colors"
      style={{
        flex: expanded ? 2 : 1,
        background: "#fff",
        border: expanded ? "1px solid #DDD6FE" : "1px solid #EEEEEE",
        borderRadius: 10,
        padding: expanded ? 0 : "20px 16px",
        display: "flex",
        flexDirection: "column",
        alignItems: expanded ? "stretch" : "center",
        gap: expanded ? 0 : 8,
        textDecoration: "none",
        transition: "flex 200ms ease, border-color 200ms",
        overflow: "hidden",
      }}
    >
      {!expanded && (
        <>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span style={{ fontSize: 13, fontWeight: 500, color: "#333333" }}>Python SDK</span>
          <span style={{ fontSize: 11, color: "#999999" }}>pip install cognee</span>
        </>
      )}

      {expanded && (
        <div onClick={(e) => e.stopPropagation()} style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {/* Header */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "16px 20px", borderBottom: "1px solid #E4E4E7" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M7 8l5 4 5-4M7 16l5-4 5 4" stroke="#6510F4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span style={{ fontSize: 14, fontWeight: 600, color: "#18181B" }}>Connect with Python SDK</span>
            <button onClick={() => setExpanded(false)} className="cursor-pointer" style={{ marginLeft: "auto", background: "none", border: "none", color: "#A1A1AA", fontSize: 16, padding: 2 }}>&#10005;</button>
          </div>

          {/* Steps */}
          <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Step 1 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>1</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Install the SDK</span>
            </div>
            <div className="flex items-center justify-between" style={{ background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
              <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>pip install cognee</span>
              <CopyButton text="pip install cognee" />
            </div>

            <div style={{ height: 1, background: "#E4E4E7" }} />

            {/* Step 2 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>2</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Set your API key</span>
            </div>
            <div className="flex items-center justify-between" style={{ background: "#18181B", borderRadius: 8, padding: "10px 16px" }}>
              <span style={{ fontSize: 13, color: "#A1A1AA", fontFamily: '"Fira Code", monospace' }}>export COGNEE_API_KEY=your-key</span>
              <CopyButton text="export COGNEE_API_KEY=your-key" />
            </div>

            <div style={{ height: 1, background: "#E4E4E7" }} />

            {/* Step 3 */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="flex items-center justify-center flex-shrink-0 rounded-full" style={{ width: 24, height: 24, background: "#F0EDFF" }}>
                <span style={{ color: "#6510F4", fontSize: 12, fontWeight: 600 }}>3</span>
              </div>
              <span style={{ fontSize: 13, fontWeight: 500, color: "#18181B" }}>Connect to your instance</span>
            </div>
            <div style={{ background: "#18181B", borderRadius: 8, padding: "12px 16px", position: "relative" }}>
              <pre style={{ margin: 0, fontSize: 12, lineHeight: "20px", fontFamily: '"Fira Code", monospace', color: "#A1A1AA" }}>
{`import cognee

cognee.config.set_llm_api_key("your-key")

await cognee.add("Your text data")
await cognee.cognify()

results = await cognee.search("query")
print(results)`}
              </pre>
              <span style={{ position: "absolute", top: 12, right: 16 }}>
                <CopyButton text={`import cognee\n\ncognee.config.set_llm_api_key("your-key")\n\nawait cognee.add("Your text data")\nawait cognee.cognify()\n\nresults = await cognee.search("query")\nprint(results)`} />
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Dev: onboarding toggle */}
      <div style={{ display: "flex", gap: 8, paddingTop: 8 }}>
        <Link href="/onboarding" style={{ fontSize: 12, color: "#A1A1AA", textDecoration: "none" }}>
          Open onboarding
        </Link>
        <span style={{ color: "#E4E4E7" }}>|</span>
        <Link href="/onboarding?source=serve" style={{ fontSize: 12, color: "#A1A1AA", textDecoration: "none" }}>
          Serve onboarding
        </Link>
      </div>
    </div>
  );
}

// ── Icons ──

function UploadIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 16V8M12 8L8 12M12 8L16 12" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
function KeyNavIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="4" y="8" width="16" height="8" rx="2" stroke="#6C47FF" strokeWidth="1.5" /><circle cx="9" cy="12" r="1.5" fill="#6C47FF" /><path d="M14 10h3M14 14h3" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
function AgentNavIcon() { return <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="#6C47FF" strokeWidth="1.5" /><path d="M5.5 21a6.5 6.5 0 0113 0" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /><path d="M15 3l2-2M9 3L7 1" stroke="#6C47FF" strokeWidth="1.5" strokeLinecap="round" /></svg>; }
