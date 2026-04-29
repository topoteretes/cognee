/**
 * Cognee workspace MCP App.
 * Layout: AgentBar (top) + DatasetRail (left sidebar) + Main column
 * containing Composer, ResultPanel, and graph iframe wrapped in a stage.
 */
import "./design.css";
// d3 ships a UMD bundle but its package.json `exports` doesn't expose it
// under any modern condition Vite resolves with, so we read the file
// directly from node_modules. The relative path is stable (npm install
// always lands d3 here) and avoids a separate copy step.
import d3MinJs from "../node_modules/d3/dist/d3.min.js?raw";
import type { App, McpUiHostContext } from "@modelcontextprotocol/ext-apps";
import { useApp } from "@modelcontextprotocol/ext-apps/react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { StrictMode, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const MAX_WORKSPACE_HEIGHT = 800;
const DEFAULT_WORKSPACE_HEIGHT = 720;

// Cognee's graph HTML loads d3 via <script src="https://d3js.org/d3.v7.min.js">.
// MCP App iframes apply a Content-Security-Policy that blocks external script
// loads, so we substitute the CDN tag with d3 inlined from our npm dependency
// before handing the HTML to the iframe via srcDoc.
//
// Brittleness flag: the regex below hard-codes cognee's exact CDN script tag
// from cognee/modules/visualization/cognee_network_visualization.py. If cognee
// changes that tag (different version, different host, additional attributes),
// this regex won't match and the iframe stays blank. Long-term fix: bundle
// d3 inside cognee's template upstream so consumers don't need this patch.
//
// Use the function form of `replace` so JS's $-substitution rules don't fire
// inside the d3 source (which contains regex code with `$1`, `$'`, etc.).
const D3_CDN_TAG_RE = /<script\s+src="https:\/\/d3js\.org\/d3\.v7\.min\.js"\s*><\/script>/;
const D3_INLINE_SCRIPT = `<script>${d3MinJs.replace(/<\/script>/g, "<\\/script>")}</script>`;

function inlineD3(html: string): string {
  return html.replace(D3_CDN_TAG_RE, () => D3_INLINE_SCRIPT);
}

// ── Helpers ────────────────────────────────────────────────────

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function textOf(result: CallToolResult): string | undefined {
  return result.content?.find((c) => c.type === "text")?.text;
}

function htmlOf(result: CallToolResult): string | null {
  const s = result.structuredContent as { html?: string } | undefined;
  return typeof s?.html === "string" ? s.html : null;
}

interface Dataset {
  id: string;
  name: string;
}

interface DataItem {
  id: string;
  name: string;
}

function datasetsOf(result: CallToolResult): Dataset[] {
  const s = result.structuredContent as { datasets?: Dataset[] } | undefined;
  return Array.isArray(s?.datasets) ? s.datasets : [];
}

function dataItemsOf(result: CallToolResult): DataItem[] {
  const s = result.structuredContent as { data?: DataItem[] } | undefined;
  return Array.isArray(s?.data) ? s.data : [];
}

function resolveWrapperHeight(hostContext: McpUiHostContext | undefined): number {
  const dims = hostContext?.containerDimensions as
    | { height?: number; maxHeight?: number }
    | undefined;
  if (typeof dims?.height === "number") return Math.min(dims.height, MAX_WORKSPACE_HEIGHT);
  if (typeof dims?.maxHeight === "number") return Math.min(dims.maxHeight, MAX_WORKSPACE_HEIGHT);
  if (hostContext?.displayMode === "fullscreen") return MAX_WORKSPACE_HEIGHT;
  return DEFAULT_WORKSPACE_HEIGHT;
}

// ── Inline icons (Lucide-style, 24x24 viewBox, stroke 1.8) ─────

const ICON_PATHS = {
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  x: <path d="M18 6 6 18M6 6l12 12" />,
  caret: <path d="m9 6 6 6-6 6" />,
  upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </>
  ),
  text: <path d="M4 7V5h16v2M9 5v14M15 19h-6" />,
  refresh: (
    <>
      <path d="M3 12a9 9 0 0 1 15.5-6.5L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-15.5 6.5L3 16" />
      <path d="M3 21v-5h5" />
    </>
  ),
  doc: (
    <>
      <path d="M14 3v5h5" />
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    </>
  ),
  send: (
    <>
      <path d="m22 2-7 20-4-9-9-4z" />
      <path d="M22 2 11 13" />
    </>
  ),
};

function Icon({
  name,
  size = 14,
  className,
}: {
  name: keyof typeof ICON_PATHS;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {ICON_PATHS[name]}
    </svg>
  );
}

// ── Cognee mark (5 pillars) ─────────────────────────────────────

function CogneeMark({ size = 16 }: { size?: number }) {
  const heights = [0.45, 0.85, 1, 0.7, 0.55];
  return (
    <span
      style={{
        height: size,
        display: "inline-flex",
        alignItems: "flex-end",
        gap: 1.5,
        color: "var(--cg-purple)",
      }}
    >
      {heights.map((h, i) => (
        <span
          key={i}
          style={{
            width: 2,
            height: `${h * size}px`,
            background: "currentColor",
            borderRadius: 1,
          }}
        />
      ))}
    </span>
  );
}

// ── Agent bar ────────────────────────────────────────────────────

function AgentBar({
  agentName,
  defaultDataset,
  agentScoped,
}: {
  agentName: string;
  defaultDataset: string | null;
  agentScoped: boolean;
}) {
  return (
    <div className="agentbar">
      <CogneeMark size={14} />
      {agentName ? (
        <span className="ab-pill">
          <span className="dot" />
          {agentName}
        </span>
      ) : (
        <span className="ab-pill muted">connecting…</span>
      )}
      <div className="ab-sep" />
      <span className="ab-meta">
        Default dataset:{" "}
        <strong>{defaultDataset ?? "—"}</strong>
      </span>
      {!agentScoped && <span className="ab-pill muted">scoping off</span>}
      <div className="ab-spacer" />
    </div>
  );
}

// ── Dataset rail (sidebar) ──────────────────────────────────────

function DatasetRail({
  app,
  datasets,
  selectedDataset,
  disabled,
  dataRefreshToken,
  onSelectDataset,
  onDatasetsChanged,
  setStatus,
  setBusy,
}: {
  app: App;
  datasets: Dataset[];
  selectedDataset: string | null;
  disabled: boolean;
  dataRefreshToken: number;
  onSelectDataset: (name: string | null) => void;
  onDatasetsChanged: () => Promise<void>;
  setStatus: (s: string) => void;
  setBusy: (b: boolean) => void;
}) {
  const [confirmingDataset, setConfirmingDataset] = useState<string | null>(null);
  const [confirmingItem, setConfirmingItem] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [dataItems, setDataItems] = useState<Record<string, DataItem[]>>({});
  const [loadingData, setLoadingData] = useState<Set<string>>(new Set());

  const loadDataItems = async (datasetId: string) => {
    setLoadingData((s) => new Set(s).add(datasetId));
    try {
      const result = await app.callServerTool({
        name: "list_dataset_data_json",
        arguments: { dataset_id: datasetId },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "failed to load data items"}`);
        return;
      }
      setDataItems((prev) => ({ ...prev, [datasetId]: dataItemsOf(result) }));
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingData((s) => {
        const next = new Set(s);
        next.delete(datasetId);
        return next;
      });
    }
  };

  const toggleExpand = (datasetId: string) => {
    setExpanded((s) => {
      const next = new Set(s);
      if (next.has(datasetId)) {
        next.delete(datasetId);
      } else {
        next.add(datasetId);
        void loadDataItems(datasetId);
      }
      return next;
    });
  };

  useEffect(() => {
    if (dataRefreshToken === 0) return;
    setDataItems({});
    expanded.forEach((id) => void loadDataItems(id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataRefreshToken]);

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setStatus(`Creating '${name}'...`);
    try {
      const result = await app.callServerTool({
        name: "create_dataset_json",
        arguments: { name },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "create failed"}`);
        return;
      }
      setNewName("");
      await onDatasetsChanged();
      onSelectDataset(name);
      setStatus(`Created '${name}'.`);
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteDataset = async (name: string) => {
    setBusy(true);
    setStatus(`Deleting '${name}'...`);
    try {
      const result = await app.callServerTool({
        name: "delete_dataset",
        arguments: { dataset_name: name },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "delete failed"}`);
        return;
      }
      setStatus(textOf(result) ?? `Deleted '${name}'.`);
      setConfirmingDataset(null);
      if (selectedDataset === name) onSelectDataset(null);
      await onDatasetsChanged();
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteItem = async (datasetId: string, dataId: string) => {
    setBusy(true);
    setStatus("Deleting data item...");
    try {
      const result = await app.callServerTool({
        name: "delete",
        arguments: { data_id: dataId, dataset_id: datasetId },
      });
      if (result.isError) {
        setStatus(`Error: ${textOf(result) ?? "delete failed"}`);
        return;
      }
      setStatus(textOf(result) ?? "Data item deleted.");
      setConfirmingItem(null);
      await loadDataItems(datasetId);
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="rail">
      <div className="rail-section">
        <div className="rail-label">
          <span>Datasets</span>
          <span className="count">{datasets.length}</span>
        </div>
      </div>
      <div className="rail-datasets">
        {datasets.length === 0 && (
          <div style={{ padding: 8, fontSize: 12, color: "var(--cg-fg-muted)" }}>
            No datasets yet.
          </div>
        )}
        {datasets.map((d) => {
          const isActive = selectedDataset === d.name;
          const isExpanded = expanded.has(d.id);
          const items = dataItems[d.id];
          const isLoading = loadingData.has(d.id);
          if (confirmingDataset === d.name) {
            return (
              <div key={d.id} className="ds-confirm">
                <span className="label">Delete "{d.name}"?</span>
                <button
                  className="yes"
                  type="button"
                  onClick={() => handleDeleteDataset(d.name)}
                  disabled={disabled}
                >
                  Delete
                </button>
                <button
                  className="no"
                  type="button"
                  onClick={() => setConfirmingDataset(null)}
                >
                  Cancel
                </button>
              </div>
            );
          }
          return (
            <div key={d.id}>
              <div
                className={`ds-row${isActive ? " active" : ""}${
                  isExpanded ? " expanded" : ""
                }`}
              >
                <button
                  type="button"
                  className="ds-caret"
                  onClick={() => toggleExpand(d.id)}
                  aria-label={isExpanded ? "Collapse" : "Expand"}
                  aria-expanded={isExpanded}
                >
                  <Icon name="caret" size={14} />
                </button>
                <button
                  type="button"
                  className="ds-name"
                  onClick={() => onSelectDataset(d.name)}
                  aria-pressed={isActive}
                  title={`Select ${d.name}`}
                >
                  {d.name}
                </button>
                <button
                  type="button"
                  className="ds-x"
                  title={`Delete ${d.name}`}
                  onClick={() => setConfirmingDataset(d.name)}
                >
                  <Icon name="x" size={12} />
                </button>
              </div>
              {isExpanded && (
                <div className="ds-children">
                  {isLoading && (
                    <div className="ds-item empty">Loading…</div>
                  )}
                  {!isLoading && items && items.length === 0 && (
                    <div className="ds-item empty">empty</div>
                  )}
                  {!isLoading &&
                    items &&
                    items.map((item) => {
                      const rowKey = `${d.id}:${item.id}`;
                      if (confirmingItem === rowKey) {
                        return (
                          <div key={item.id} className="di-confirm">
                            <span className="label">Delete "{item.name}"?</span>
                            <button
                              className="yes"
                              type="button"
                              onClick={() => handleDeleteItem(d.id, item.id)}
                              disabled={disabled}
                            >
                              Delete
                            </button>
                            <button
                              className="no"
                              type="button"
                              onClick={() => setConfirmingItem(null)}
                            >
                              Cancel
                            </button>
                          </div>
                        );
                      }
                      return (
                        <div key={item.id} className="ds-item">
                          <Icon name="doc" size={12} className="di-icon" />
                          <span className="di-name" title={item.name}>
                            {item.name}
                          </span>
                          <button
                            type="button"
                            className="di-x"
                            title={`Delete ${item.name}`}
                            onClick={() => setConfirmingItem(rowKey)}
                          >
                            <Icon name="x" size={11} />
                          </button>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="ds-create">
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleCreate();
          }}
          placeholder="New dataset"
          disabled={disabled}
        />
        <button
          type="button"
          onClick={handleCreate}
          disabled={disabled || !newName.trim()}
          aria-label="Create dataset"
        >
          <Icon name="plus" size={12} />
        </button>
      </div>
    </aside>
  );
}

// ── Composer ────────────────────────────────────────────────────

const SEARCH_TYPES = [
  { id: "GRAPH_COMPLETION", label: "Graph" },
  { id: "RAG_COMPLETION", label: "RAG" },
  { id: "CHUNKS", label: "Chunks" },
];

function Composer({
  app,
  selectedDataset,
  busy,
  setBusy,
  status,
  setStatus,
  onGraphRefresh,
  onSearchResult,
  onIngestSucceeded,
}: {
  app: App;
  selectedDataset: string | null;
  busy: boolean;
  setBusy: (b: boolean) => void;
  status: string;
  setStatus: (s: string) => void;
  onGraphRefresh: () => Promise<void>;
  onSearchResult: (entry: SearchEntry | null) => void;
  onIngestSucceeded: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [text, setText] = useState("");
  const [showText, setShowText] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchType, setSearchType] = useState("GRAPH_COMPLETION");

  const handleSearch = async () => {
    const q = searchQuery.trim();
    if (!q) return;
    setBusy(true);
    setStatus("");
    onSearchResult({
      type: searchType,
      query: q,
      dataset: selectedDataset ?? "—",
      body: "Searching…",
    });
    try {
      const args: Record<string, string> = {
        search_query: q,
        search_type: searchType,
      };
      if (selectedDataset) args.datasets = selectedDataset;
      const result = await app.callServerTool({ name: "search", arguments: args });
      const msg = textOf(result) ?? JSON.stringify(result.content);
      onSearchResult({
        type: searchType,
        query: q,
        dataset: selectedDataset ?? "—",
        body: result.isError ? `Error: ${msg}` : msg,
      });
    } catch (err) {
      onSearchResult({
        type: searchType,
        query: q,
        dataset: selectedDataset ?? "—",
        body: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setBusy(false);
    }
  };

  const handleAddText = async () => {
    const data = text.trim();
    if (!data) return;
    setBusy(true);
    setStatus("Adding text…");
    try {
      const args: Record<string, string> = { data };
      if (selectedDataset) args.dataset_name = selectedDataset;
      const result = await app.callServerTool({ name: "cognify", arguments: args });
      const msg = textOf(result) ?? JSON.stringify(result.content);
      setStatus(result.isError ? `Error: ${msg}` : msg);
      if (!result.isError) {
        setText("");
        onIngestSucceeded();
      }
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatus(`"${file.name}" is ${(file.size / 1024 / 1024).toFixed(1)} MB (10 MB max).`);
      return;
    }
    setBusy(true);
    setStatus(`Uploading ${file.name}…`);
    try {
      const content_base64 = arrayBufferToBase64(await file.arrayBuffer());
      const args: Record<string, string> = { filename: file.name, content_base64 };
      if (selectedDataset) args.dataset_name = selectedDataset;
      const result = await app.callServerTool({
        name: "cognify_file",
        arguments: args,
      });
      const msg = textOf(result) ?? JSON.stringify(result.content);
      setStatus(result.isError ? `Error: ${msg}` : msg);
      if (!result.isError) onIngestSucceeded();
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="composer">
      <div className="composer-row">
        <div className="composer-search">
          <Icon name="search" size={14} className="icn" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleSearch();
            }}
            placeholder={
              selectedDataset
                ? `Ask anything about ${selectedDataset}…`
                : "Ask a question…"
            }
            disabled={busy}
          />
          <div className="types">
            {SEARCH_TYPES.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`search-type-btn${searchType === t.id ? " active" : ""}`}
                onClick={() => setSearchType(t.id)}
                title={t.id}
              >
                {t.label}
              </button>
            ))}
            <span className="types-divider" />
            <button
              type="button"
              className="go-btn"
              onClick={handleSearch}
              disabled={busy || !searchQuery.trim()}
            >
              <Icon name="send" size={11} />
              Recall
            </button>
          </div>
        </div>
      </div>

      <div className="composer-row">
        <button
          type="button"
          className="cmpsr-btn primary"
          onClick={() => fileRef.current?.click()}
          disabled={busy}
        >
          <Icon name="upload" />
          Cognify file
        </button>
        <input
          ref={fileRef}
          type="file"
          style={{ display: "none" }}
          onChange={handleFile}
        />
        <button
          type="button"
          className="cmpsr-btn"
          onClick={() => setShowText((s) => !s)}
          disabled={busy}
        >
          <Icon name="text" />
          {showText ? "Hide text" : "Add text"}
        </button>
        <button
          type="button"
          className="cmpsr-btn"
          onClick={() => void onGraphRefresh()}
          disabled={busy}
        >
          <Icon name="refresh" />
          Refresh
        </button>
        <span className="cmpsr-context">
          target → <strong>{selectedDataset ?? "—"}</strong>
        </span>
      </div>

      {showText && (
        <div className="composer-row" style={{ alignItems: "stretch" }}>
          <textarea
            className="cmpsr-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste or type text to cognify into the selected dataset…"
            disabled={busy}
          />
          <button
            type="button"
            className="cmpsr-btn primary"
            onClick={handleAddText}
            disabled={busy || !text.trim()}
            style={{ alignSelf: "flex-end" }}
          >
            Cognify text
          </button>
        </div>
      )}

      {status && <div className="cmpsr-status">{status}</div>}
    </div>
  );
}

// ── Result panel ────────────────────────────────────────────────

interface SearchEntry {
  type: string;
  query: string;
  dataset: string;
  body: string;
}

function ResultPanel({
  entry,
  onClose,
}: {
  entry: SearchEntry | null;
  onClose: () => void;
}) {
  if (!entry) return null;
  return (
    <div className="result">
      <div className="answer-rail" />
      <div className="answer-body">
        <div className="answer-eyebrow">
          {entry.type}
          <span className="qmeta">· {entry.dataset}</span>
        </div>
        <div className="answer-text">
          <strong style={{ color: "var(--cg-fg-dark)", fontWeight: 500 }}>
            {entry.query}
          </strong>
          {"\n\n"}
          {entry.body}
        </div>
      </div>
      <button
        type="button"
        className="result-close"
        onClick={onClose}
        aria-label="Close search result"
      >
        <Icon name="x" size={13} />
      </button>
    </div>
  );
}

// ── Graph stage ─────────────────────────────────────────────────

function GraphStage({
  html,
  status,
  busy,
}: {
  html: string | null;
  status: string;
  busy: boolean;
}) {
  // d3 source is ~280 KB; recompute the inlined HTML only when the upstream
  // graph HTML actually changes, not on every parent re-render.
  const srcDoc = useMemo(() => (html ? inlineD3(html) : null), [html]);
  return (
    <div className="graph-wrap">
      <div className="gtabs">
        <button type="button" className="gtab active">
          <span>Graph</span>
        </button>
      </div>
      <div className="gstage">
        {srcDoc ? (
          <iframe
            title="Cognee Knowledge Graph"
            srcDoc={srcDoc}
            sandbox="allow-scripts"
          />
        ) : (
          <div className="empty-state">
            {status || "No graph data yet. Cognify content to populate it."}
          </div>
        )}
        <div className={`status-pill${busy ? " processing" : ""}`}>
          <span className="dot" />
          {busy ? "Cognifying…" : html ? "live" : "idle"}
        </div>
      </div>
    </div>
  );
}

// ── Workspace root ──────────────────────────────────────────────

function WorkspaceApp() {
  const [hostContext, setHostContext] = useState<McpUiHostContext | undefined>();
  const [graphHtml, setGraphHtml] = useState<string | null>(null);
  const [graphStatus, setGraphStatus] = useState<string>("Loading graph…");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [searchEntry, setSearchEntry] = useState<SearchEntry | null>(null);
  const [clientName, setClientName] = useState<string>("");
  const [agentDefaultDataset, setAgentDefaultDataset] = useState<string | null>(null);
  const [agentScoped, setAgentScoped] = useState<boolean>(true);
  const [composerStatus, setComposerStatus] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [dataRefreshToken, setDataRefreshToken] = useState<number>(0);

  const { app, error } = useApp({
    appInfo: { name: "Cognee Workspace", version: "0.2.0" },
    capabilities: {},
    onAppCreated: (app) => {
      app.ontoolresult = async (result) => {
        const html = htmlOf(result);
        if (html) {
          setGraphHtml(html);
          setGraphStatus("");
        }
      };
      app.onhostcontextchanged = (params) =>
        setHostContext((prev) => ({ ...prev, ...params }));
      app.onerror = console.error;
    },
  });

  useEffect(() => {
    if (app) setHostContext(app.getHostContext());
  }, [app]);

  const loadGraph = async (a: App, datasetName: string | null = selectedDataset) => {
    setGraphStatus("Loading graph…");
    try {
      const result = await a.callServerTool({
        name: "visualize_graph_ui",
        arguments: datasetName ? { dataset_name: datasetName } : {},
      });
      if (result.isError) {
        setGraphStatus(`Error: ${textOf(result) ?? "failed to load graph"}`);
        return;
      }
      const html = htmlOf(result);
      if (html) {
        setGraphHtml(html);
        setGraphStatus("");
      } else {
        setGraphHtml(null);
        setGraphStatus("No graph data yet. Cognify content to populate it.");
      }
    } catch (err) {
      setGraphStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const reloadDatasets = async (preferredOverride?: string | null) => {
    if (!app) return;
    try {
      const result = await app.callServerTool({
        name: "list_datasets_json",
        arguments: {},
      });
      if (result.isError) return;
      const list = datasetsOf(result);
      setDatasets(list);
      setSelectedDataset((prev) => {
        if (prev && list.some((d) => d.name === prev)) return prev;
        if (list.length === 0) return null;
        const preferred =
          preferredOverride !== undefined ? preferredOverride : agentDefaultDataset;
        const match =
          (preferred ? list.find((d) => d.name === preferred) : undefined) ??
          list.find((d) => d.name === "main_dataset") ??
          list[0];
        return match.name;
      });
    } catch {
      /* ignore; rail just stays empty */
    }
  };

  useEffect(() => {
    if (!app) return;
    (async () => {
      let preferred: string | null = null;
      try {
        const info = await app.callServerTool({
          name: "get_client_info_json",
          arguments: {},
        });
        if (!info.isError) {
          const s = info.structuredContent as
            | {
                client?: { name?: string };
                default_dataset?: string;
                agent_scoped?: boolean;
              }
            | undefined;
          if (s?.client?.name) setClientName(s.client.name);
          if (typeof s?.agent_scoped === "boolean") setAgentScoped(s.agent_scoped);
          if (s?.default_dataset) {
            setAgentDefaultDataset(s.default_dataset);
            preferred = s.default_dataset;
          }
        }
      } catch {
        /* ignore; bar just stays empty */
      }
      await reloadDatasets(preferred);
      // Pass `preferred` explicitly: setSelectedDataset from reloadDatasets
      // hasn't propagated into loadGraph's closure yet on this mount tick.
      await loadGraph(app, preferred);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [app]);

  const wrapperStyle: React.CSSProperties = {
    width: "100%",
    height: resolveWrapperHeight(hostContext),
  };

  if (error) {
    return (
      <div className="cg-root ws" style={wrapperStyle}>
        <div style={{ padding: 16, color: "var(--cg-status-error)" }}>
          ERROR: {error.message}
        </div>
      </div>
    );
  }
  if (!app) {
    return (
      <div className="cg-root ws" style={wrapperStyle}>
        <div style={{ padding: 16, color: "var(--cg-fg-muted)" }}>Connecting…</div>
      </div>
    );
  }

  return (
    <div className="cg-root ws" style={wrapperStyle}>
      <AgentBar
        agentName={clientName}
        defaultDataset={agentDefaultDataset}
        agentScoped={agentScoped}
      />
      <div className="ws-body">
        <DatasetRail
          app={app}
          datasets={datasets}
          selectedDataset={selectedDataset}
          disabled={busy}
          dataRefreshToken={dataRefreshToken}
          onSelectDataset={setSelectedDataset}
          onDatasetsChanged={reloadDatasets}
          setStatus={setComposerStatus}
          setBusy={setBusy}
        />
        <main className="main">
          <Composer
            app={app}
            selectedDataset={selectedDataset}
            busy={busy}
            setBusy={setBusy}
            status={composerStatus}
            setStatus={setComposerStatus}
            onGraphRefresh={() => loadGraph(app)}
            onSearchResult={setSearchEntry}
            onIngestSucceeded={() => setDataRefreshToken((t) => t + 1)}
          />
          <ResultPanel entry={searchEntry} onClose={() => setSearchEntry(null)} />
          <GraphStage html={graphHtml} status={graphStatus} busy={busy} />
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <WorkspaceApp />
  </StrictMode>,
);
